// Contact Luck -- Play Explorer (redesign Phase 5).
//
// Purely presentational, player-first, and sharded. The fetch contract from
// Phase 4.2 is unchanged and must stay that way:
//   1. On page load, fetch ONLY the small players.json catalog -- never any
//      play data yet.
//   2. ONLY after a hitter is selected, fetch that ONE hitter's
//      players/<batter_id>.json index -- never every batter's file, never a
//      monolithic index (at full 2024 development-data scale that measured
//      ~29.3 MiB, over Cloudflare Pages' 25 MiB per-asset limit).
//   3. Filter and sort that one hitter's already-loaded plays client-side.
//      Every value rendered, Contact Luck included, is copied verbatim from
//      the fetched row. Nothing here is recomputed.
//
// Phase 5 changes three things and nothing else about that contract:
//   * The picker is the shared ARIA 1.2 combobox from app.js, not a second
//     weaker implementation.
//   * THE URL IS THE SINGLE SOURCE OF TRUTH for which hitter is selected.
//     Selecting pushes `?batter=<id>`; popstate re-derives the selection
//     from the address. The in-memory selection is never the authority --
//     that is what keeps Back/Forward honest and the URL never stale.
//   * Each play is drawn on the build-supplied `run_value` domain. The
//     domain arrives as data; this file interpolates a layout percentage
//     against it and derives no scale of its own.
(function () {
  "use strict";

  var PLAYERS_URL = "players.json";
  var SHOWCASE_URL = "showcase.json";

  var OUTCOME_LABELS = {
    out: "Out",
    single: "Single",
    double: "Double",
    triple: "Triple",
    home_run: "Home run",
  };

  //: How many plays each showcase group shows before its one-way reveal.
  var SHOWCASE_INITIAL_VISIBLE = 6;

  var EM_DASH = "—";
  var MINUS = "−";

  function qs(selector, root) {
    return (root || document).querySelector(selector);
  }

  function formatSigned(value, digits) {
    if (value === null || value === undefined) return EM_DASH;
    // Redesign Phase 1 numeric primitive: explicit sign, and U+2212 MINUS
    // SIGN rather than ASCII hyphen-minus, so signed values align in a
    // tabular-figure column and read identically to the build-time `signed`
    // Jinja filter in dashboard/build.py.
    var text = value.toFixed(digits === undefined ? 2 : digits);
    return (value >= 0 ? "+" : "") + text.replace("-", MINUS);
  }

  function outcomeLabel(cls) {
    return OUTCOME_LABELS[cls] || cls;
  }

  function playerLabel(entry) {
    return entry.batter_name || "Player " + entry.batter_id;
  }

  function plural(n, word) {
    return n + " " + word + (n === 1 ? "" : "s");
  }

  function evLaText(launchSpeed, launchAngle) {
    // In the results table's own column a missing measurement is an em
    // dash, which is what a numeric column uses for "not recorded". In
    // prose it is left out entirely: a sentence made of dashes reads as a
    // rendering fault rather than as absent data.
    var parts = [];
    if (launchSpeed !== null && launchSpeed !== undefined) {
      parts.push(launchSpeed.toFixed(1) + " mph");
    }
    if (launchAngle !== null && launchAngle !== undefined) {
      parts.push(Math.round(launchAngle) + "°");
    }
    return parts.join(" · ");
  }

  function playUrl(playId) {
    // Query-param route (Phase 4.1: a single static /plays/ shell, never one
    // directory per play_id). Absolute path, matching this site's routing
    // convention (root_prefix is always "/").
    return "/plays/?id=" + encodeURIComponent(playId);
  }

  // ── The run-value scale ────────────────────────────────────────────────
  // The domain is computed at BUILD time over every published play and
  // delivered as JSON (see dashboard/build.py). Mapping a value to a
  // fraction of the plot field is a layout calculation against that fixed
  // domain -- not a score, rank, interval or probability -- and the domain
  // is never re-derived here from whichever hitter happens to be loaded.
  var scale = null;

  function loadScale() {
    var el = document.getElementById("explore-run-value-scale");
    if (!el) return null;
    try {
      var raw = JSON.parse(el.textContent);
      if (raw.domain_max <= raw.domain_min) return null;
      return raw;
    } catch (err) {
      return null;
    }
  }

  function scalePct(value) {
    var span = scale.domain_max - scale.domain_min;
    var fraction = (value - scale.domain_min) / span;
    if (fraction < 0) fraction = 0;
    if (fraction > 1) fraction = 1;
    return (fraction * 100).toFixed(4) + "%";
  }

  function signClass(value) {
    return value >= 0 ? "cl-scale-favorable" : "cl-scale-unfavorable";
  }

  // ── Showcase ───────────────────────────────────────────────────────────
  // A ruled list, not a card grid: the baseline drew bordered cards with a
  // coloured top rule, spending the sign colours on decoration. Here the
  // colour is on the mark and the numeral, where it encodes the sign.
  function buildShowcaseItem(row) {
    var li = document.createElement("li");
    li.className = "explore-showcase-item";

    var head = document.createElement("p");
    head.className = "explore-showcase-item-head";
    var link = document.createElement("a");
    link.className = "explore-showcase-item-link";
    link.href = playUrl(row.play_id);
    link.textContent = row.batter_name || "Player " + row.batter_id;
    head.appendChild(link);
    var date = document.createElement("span");
    date.className = "explore-showcase-item-date";
    date.textContent = row.game_date;
    head.appendChild(date);
    li.appendChild(head);

    var detail = document.createElement("p");
    detail.className = "explore-showcase-item-detail";
    var contact = evLaText(row.launch_speed, row.launch_angle);
    detail.textContent =
      [outcomeLabel(row.outcome_class), contact].filter(Boolean).join(" · ") +
      " · expected " +
      formatSigned(row.expected_run_value) +
      ", actual " +
      formatSigned(row.observed_run_value);
    li.appendChild(detail);

    if (scale) {
      var field = document.createElement("div");
      field.className = "cl-scale-field explore-play-field " + signClass(row.contact_luck_runs);
      field.setAttribute("aria-hidden", "true");
      field.style.setProperty("--cl-pt", scalePct(row.contact_luck_runs));
      var dot = document.createElement("span");
      dot.className = "cl-scale-point";
      field.appendChild(dot);
      li.appendChild(field);
    }

    var value = document.createElement("p");
    value.className = "explore-showcase-item-value num " + signClass(row.contact_luck_runs);
    value.textContent = formatSigned(row.contact_luck_runs);
    li.appendChild(value);

    if (row.interactive_available) {
      var note = document.createElement("p");
      note.className = "explore-showcase-item-note";
      note.textContent = "What if? available";
      li.appendChild(note);
    }
    return li;
  }

  function renderShowcaseGroup(section, groupKey, rows) {
    var listEl = qs('[data-role="showcase-cards-' + groupKey + '"]', section);
    var revealBtn = qs('[data-role="showcase-reveal-' + groupKey + '"]', section);
    if (!listEl) return;

    var sorted = rows.slice().sort(function (a, b) {
      return a.rank - b.rank;
    });
    var rest = sorted.slice(SHOWCASE_INITIAL_VISIBLE);
    sorted.slice(0, SHOWCASE_INITIAL_VISIBLE).forEach(function (row) {
      listEl.appendChild(buildShowcaseItem(row));
    });

    // Preserved one-way reveal: the trigger hides after use, and nothing
    // already on screen moves or re-orders.
    if (rest.length && revealBtn) {
      revealBtn.hidden = false;
      revealBtn.textContent = "Show all " + sorted.length;
      revealBtn.addEventListener("click", function () {
        var firstNew = null;
        rest.forEach(function (row) {
          var el = buildShowcaseItem(row);
          if (!firstNew) firstNew = el;
          listEl.appendChild(el);
        });
        revealBtn.hidden = true;
        // Focus would otherwise land on <body> when the trigger vanishes.
        if (firstNew) {
          var target = qs("a", firstNew);
          if (target) target.focus();
        }
      });
    }
  }

  function initShowcaseSection() {
    var section = qs('[data-role="showcase-section"]');
    if (!section) return;

    var loadingEl = qs('[data-role="showcase-loading"]', section);
    var bodyEl = qs('[data-role="showcase-body"]', section);

    fetch(SHOWCASE_URL)
      .then(function (response) {
        if (!response.ok) throw new Error("failed to load showcase: " + response.status);
        return response.json();
      })
      .then(function (rows) {
        renderShowcaseGroup(
          section,
          "favorable",
          rows.filter(function (r) {
            return r.group === "favorable";
          })
        );
        renderShowcaseGroup(
          section,
          "unfavorable",
          rows.filter(function (r) {
            return r.group === "unfavorable";
          })
        );
        if (loadingEl) loadingEl.hidden = true;
        if (bodyEl) bodyEl.hidden = false;
      })
      .catch(function (err) {
        if (loadingEl) {
          loadingEl.textContent = "These plays could not load. The rest of the page still works.";
        }
        if (window.console && window.console.error) window.console.error(err);
      });
  }

  // ── The hitter's plays ─────────────────────────────────────────────────
  function initExplorePage() {
    var selectedRoot = qs('[data-role="explore-selected"]');
    if (!selectedRoot) return;

    var pickerWrap = qs('[data-role="explore-picker"]');
    var searchInput = qs('[data-role="explore-player-search"]');
    var listbox = qs('[data-role="explore-player-search-results"]');
    var searchStatus = qs('[data-role="explore-player-search-status"]');
    var catalogStatusEl = qs('[data-role="explore-catalog-status"]');
    var promptEl = qs('[data-role="explore-prompt"]');

    var nameEl = qs('[data-role="explore-selected-name"]');
    var metaEl = qs('[data-role="explore-selected-meta"]');
    var playerLinkEl = qs('[data-role="explore-selected-player-link"]');

    var distributionField = qs('[data-role="explore-distribution-field"]');
    var loadingEl = qs('[data-role="explore-loading"]');
    var emptyEl = qs('[data-role="explore-empty"]');
    var noPlaysEl = qs('[data-role="explore-no-plays"]');
    var countEl = qs('[data-role="explore-result-count"]');
    var resultsStatus = qs('[data-role="explore-results-status"]');
    var resultsEl = qs('[data-role="explore-results"]');
    var captionEl = qs('[data-role="explore-results-caption"]');
    var tbody = qs('[data-role="explore-results-body"]');
    var outcomeSelect = qs('[data-role="explore-outcome-filter"]');
    var luckSelect = qs('[data-role="explore-luck-filter"]');
    var sortSelect = qs('[data-role="explore-sort"]');
    var resetBtn = qs('[data-role="explore-reset"]');

    var playersCatalog = [];
    var selectedEntry = null;
    var selectedRows = [];
    var combobox = null;

    var SORTERS = {
      most_favorable: function (a, b) {
        return b.contact_luck_runs - a.contact_luck_runs;
      },
      most_unfavorable: function (a, b) {
        return a.contact_luck_runs - b.contact_luck_runs;
      },
      newest: function (a, b) {
        if (a.game_date !== b.game_date) return a.game_date < b.game_date ? 1 : -1;
        if (a.game_pk !== b.game_pk) return b.game_pk - a.game_pk;
        return a.play_id < b.play_id ? 1 : -1;
      },
      hardest_hit: function (a, b) {
        var av = a.launch_speed === null || a.launch_speed === undefined ? -Infinity : a.launch_speed;
        var bv = b.launch_speed === null || b.launch_speed === undefined ? -Infinity : b.launch_speed;
        return bv - av;
      },
    };

    function matchesFilters(row) {
      var outcome = outcomeSelect.value;
      var luck = luckSelect.value;
      if (outcome && row.outcome_class !== outcome) return false;
      if (luck === "favorable" && !(row.contact_luck_runs >= 0)) return false;
      if (luck === "unfavorable" && !(row.contact_luck_runs < 0)) return false;
      return true;
    }

    function filtersActive() {
      return outcomeSelect.value !== "" || luckSelect.value !== "all";
    }

    function renderDistribution(matching) {
      if (!scale || !distributionField) return;
      var included = {};
      matching.forEach(function (row) {
        included[row.play_id] = true;
      });

      distributionField.innerHTML = "";
      selectedRows.forEach(function (row) {
        var mark = document.createElement("span");
        mark.className =
          "explore-distribution-mark " +
          signClass(row.contact_luck_runs) +
          (included[row.play_id] ? "" : " is-filtered-out");
        mark.style.left = scalePct(row.contact_luck_runs);
        distributionField.appendChild(mark);
      });

      var values = selectedRows.map(function (row) {
        return row.contact_luck_runs;
      });
      distributionField.setAttribute(
        "aria-label",
        plural(selectedRows.length, "play") +
          " from " +
          formatSigned(Math.min.apply(null, values)) +
          " to " +
          formatSigned(Math.max.apply(null, values)) +
          " runs of Contact Luck. Every value is listed in the rows below."
      );
    }

    function buildRow(row) {
      var tr = document.createElement("tr");

      // Play identity: the date is the link, because a date plus a result is
      // how a reader tells one of this hitter's batted balls from another.
      var playTd = document.createElement("th");
      playTd.setAttribute("scope", "row");
      playTd.className = "col-play";
      var link = document.createElement("a");
      link.className = "explore-play-link";
      link.href = playUrl(row.play_id);
      link.textContent = row.game_date;
      playTd.appendChild(link);
      tr.appendChild(playTd);

      var resultTd = document.createElement("td");
      resultTd.className = "col-play-result";
      resultTd.textContent = outcomeLabel(row.outcome_class);
      tr.appendChild(resultTd);

      var contactTd = document.createElement("td");
      contactTd.className = "col-play-contact num";
      contactTd.textContent = evLaText(row.launch_speed, row.launch_angle) || EM_DASH;
      tr.appendChild(contactTd);

      var expectedTd = document.createElement("td");
      expectedTd.className = "col-play-expected num";
      expectedTd.textContent = formatSigned(row.expected_run_value);
      tr.appendChild(expectedTd);

      // The verdict is ONE cell: the mark and the numeral are grid siblings
      // sharing a percentage basis, exactly as on the leaderboard, so the
      // spine at `--cl-zero` registers down every row by construction.
      var verdictTd = document.createElement("td");
      verdictTd.className = "col-play-verdict " + signClass(row.contact_luck_runs);
      if (scale) {
        var field = document.createElement("div");
        field.className = "cl-scale-field explore-play-field";
        field.setAttribute("aria-hidden", "true");
        field.style.setProperty("--cl-pt", scalePct(row.contact_luck_runs));
        var dot = document.createElement("span");
        dot.className = "cl-scale-point";
        field.appendChild(dot);
        verdictTd.appendChild(field);
      }
      var value = document.createElement("span");
      value.className = "explore-play-value num";
      value.textContent = formatSigned(row.contact_luck_runs);
      verdictTd.appendChild(value);
      tr.appendChild(verdictTd);

      return tr;
    }

    function renderResults() {
      var filtered = selectedRows.filter(matchesFilters);
      filtered.sort(SORTERS[sortSelect.value] || SORTERS.most_favorable);

      if (resetBtn) resetBtn.hidden = !filtersActive();
      renderDistribution(filtered);

      tbody.innerHTML = "";
      if (!filtered.length) {
        // Preserved invariant: the empty state hides the table AND the
        // result count together.
        resultsEl.hidden = true;
        countEl.hidden = true;
        emptyEl.hidden = false;
        if (resultsStatus) resultsStatus.textContent = "No plays match these filters.";
        return;
      }
      emptyEl.hidden = true;
      resultsEl.hidden = false;
      countEl.hidden = false;
      countEl.textContent =
        filtered.length === selectedRows.length
          ? plural(filtered.length, "play")
          : filtered.length + " of " + plural(selectedRows.length, "play");
      if (captionEl) {
        captionEl.textContent =
          "Published plays for " +
          playerLabel(selectedEntry) +
          ", with exit velocity, launch angle, recorded result and Contact Luck in runs.";
      }
      if (resultsStatus) resultsStatus.textContent = countEl.textContent + " shown.";

      var frag = document.createDocumentFragment();
      filtered.forEach(function (row) {
        frag.appendChild(buildRow(row));
      });
      tbody.appendChild(frag);
    }

    function showNoPublishedPlays() {
      resultsEl.hidden = true;
      countEl.hidden = true;
      emptyEl.hidden = true;
      noPlaysEl.hidden = false;
      noPlaysEl.textContent =
        playerLabel(selectedEntry) +
        "'s individual plays aren't currently available in Play Explorer.";
    }

    function clearSelection() {
      selectedEntry = null;
      selectedRows = [];
      selectedRoot.hidden = true;
      promptEl.hidden = false;
      searchInput.value = "";
      if (combobox) combobox.close();
    }

    function selectPlayer(entry) {
      // The ONE network request a selection makes: exactly this hitter's own
      // play index, never any other batter's file and never the catalog again.
      fetch("players/" + encodeURIComponent(String(entry.batter_id)) + ".json")
        .then(function (response) {
          if (!response.ok) throw new Error("failed to load player index: " + response.status);
          return response.json();
        })
        .then(function (rows) {
          if (!selectedEntry || selectedEntry.batter_id !== entry.batter_id) return;
          selectedRows = rows;
          loadingEl.hidden = true;
          if (!rows.length) {
            showNoPublishedPlays();
            return;
          }
          renderResults();
        })
        .catch(function (err) {
          loadingEl.textContent = "These plays could not load. Try again in a moment.";
          if (window.console && window.console.error) window.console.error(err);
        });
    }

    function showSelection(entry) {
      selectedEntry = entry;
      selectedRows = [];
      promptEl.hidden = true;
      selectedRoot.hidden = false;
      nameEl.textContent = playerLabel(entry);
      metaEl.textContent = plural(entry.play_count, "published play");
      playerLinkEl.href = "/players/" + encodeURIComponent(String(entry.batter_id)) + "/";
      searchInput.value = playerLabel(entry);

      loadingEl.hidden = false;
      loadingEl.textContent = "Loading plays…";
      resultsEl.hidden = true;
      countEl.hidden = true;
      emptyEl.hidden = true;
      noPlaysEl.hidden = true;
      tbody.innerHTML = "";

      selectPlayer(entry);
    }

    // ── URL state ────────────────────────────────────────────────────────
    // The address is the single source of truth. Every path into a selection
    // goes through applyUrl(), and the only thing a click does is change the
    // address -- which is what keeps Back and Forward honest and stops the
    // URL from ever describing a hitter who is not on screen.
    function batterIdInUrl() {
      return new URLSearchParams(window.location.search).get("batter");
    }

    function findEntry(rawId) {
      if (!rawId) return null;
      for (var i = 0; i < playersCatalog.length; i += 1) {
        if (String(playersCatalog[i].batter_id) === rawId) return playersCatalog[i];
      }
      return null;
    }

    function applyUrl() {
      var requested = batterIdInUrl();
      var entry = findEntry(requested);
      if (entry) {
        if (selectedEntry && selectedEntry.batter_id === entry.batter_id) return;
        showSelection(entry);
        return;
      }
      // An unknown or malformed id is not an error state: the normal prompt
      // stands and no request is made. A hitter can be absent from the
      // published catalog for ordinary reasons.
      clearSelection();
    }

    function pushSelection(entry) {
      var url = new URL(window.location.href);
      url.searchParams.set("batter", String(entry.batter_id));
      if (url.href !== window.location.href) {
        window.history.pushState({ batter: entry.batter_id }, "", url);
      }
      applyUrl();
    }

    window.addEventListener("popstate", applyUrl);

    [outcomeSelect, luckSelect, sortSelect].forEach(function (el) {
      el.addEventListener("change", renderResults);
    });

    if (resetBtn) {
      resetBtn.addEventListener("click", function () {
        outcomeSelect.value = "";
        luckSelect.value = "all";
        renderResults();
        outcomeSelect.focus();
      });
    }

    var factory = window.ContactLuck && window.ContactLuck.createCombobox;
    if (factory) {
      combobox = factory({
        wrap: pickerWrap,
        input: searchInput,
        listbox: listbox,
        statusEl: searchStatus,
        items: [],
        optionIdPrefix: "explore-player-option-",
        optionClassName: "explore-picker-option",
        emptyClassName: "explore-picker-empty",
        emptyMessage: function (query) {
          return "No hitter with published plays matches “" + query + "”.";
        },
        renderOption: function (li, entry) {
          var name = document.createElement("span");
          name.className = "explore-picker-option-name";
          name.textContent = playerLabel(entry);
          li.appendChild(name);
          var count = document.createElement("span");
          count.className = "explore-picker-option-count num";
          count.textContent = plural(entry.play_count, "play");
          li.appendChild(count);
          return playerLabel(entry) + ", " + plural(entry.play_count, "play");
        },
        onSelect: pushSelection,
      });
    }

    // The ONLY fetch made on page load -- the small players catalog. No
    // per-player file is ever fetched here or in a loop over the catalog;
    // each players/<batter_id>.json fetch happens exclusively inside
    // selectPlayer(), reached only through applyUrl().
    fetch(PLAYERS_URL)
      .then(function (response) {
        if (!response.ok) throw new Error("failed to load players catalog: " + response.status);
        return response.json();
      })
      .then(function (players) {
        playersCatalog = players;
        if (combobox) combobox.setItems(players);
        catalogStatusEl.hidden = true;
        searchInput.disabled = false;
        applyUrl();
      })
      .catch(function (err) {
        catalogStatusEl.textContent = "Hitters could not load. The rest of the site still works.";
        if (window.console && window.console.error) window.console.error(err);
      });
  }

  document.addEventListener("DOMContentLoaded", function () {
    scale = loadScale();
    initShowcaseSection();
    initExplorePage();
  });
})();
