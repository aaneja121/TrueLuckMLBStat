// Contact Luck v1.4.0 (Phase 4.2) -- Play Explorer landing/results page.
// Purely presentational, player-first flow:
//   1. On page load, fetch ONLY the small, already-computed players.json
//      catalog (see dashboard/explore_content.py /
//      demo/build_play_explorer_fixture.py) -- never any play data yet.
//   2. Let the visitor search/select a hitter from that catalog.
//   3. ONLY after a hitter is selected, fetch that ONE hitter's
//      players/<batter_id>.json play index -- never every batter's file,
//      never the old monolithic search-index.json (removed in Phase 4.2;
//      at full 2024 development-data scale it measured ~29.3 MiB, over
//      Cloudflare Pages' 25 MiB per-asset limit).
//   4. Filter/sort that ONE hitter's already-loaded plays entirely
//      client-side -- no per-keystroke network request, no recomputation
//      of any scoring field. Every value rendered here (including Contact
//      Luck) is copied verbatim from the fetched row.
(function () {
  "use strict";

  var PLAYERS_URL = "players.json";

  var OUTCOME_LABELS = {
    out: "Out",
    single: "Single",
    double: "Double",
    triple: "Triple",
    home_run: "Home run",
  };

  var MAX_SUGGESTIONS = 8;

  function qs(selector, root) {
    return (root || document).querySelector(selector);
  }

  function formatSigned(value) {
    if (value === null || value === undefined) return "—";
    // Redesign Phase 1 numeric primitive: explicit sign, and U+2212 MINUS
    // SIGN rather than ASCII hyphen-minus, so signed Contact Luck / run
    // values align in a tabular-figure column and read identically to the
    // build-time `signed` Jinja filter in dashboard/build.py.
    var sign = value >= 0 ? "+" : "";
    return sign + value.toFixed(2).replace("-", "\u2212");
  }

  function formatDate(iso) {
    // iso is already YYYY-MM-DD -- rendered as-is, no timezone conversion
    // (this repository never infers a timezone for a bare date string).
    return iso;
  }

  function outcomeLabel(cls) {
    return OUTCOME_LABELS[cls] || cls;
  }

  function playerLabel(entry) {
    return entry.batter_name || "Player " + entry.batter_id;
  }

  // Search matching only -- NEVER used for display (playerLabel/DOM text
  // always renders the original, accented entry.batter_name verbatim).
  // Case-insensitive, diacritic-insensitive (Unicode NFD decomposes an
  // accented character into a base letter plus a separate combining-mark
  // codepoint, e.g. U+00E9 "e-acute" -> "e" + U+0301; \u0300-\u036f is the
  // Unicode "Combining Diacritical Marks" block, stripped here so the base
  // letter is all that remains), and whitespace-normalized (trims both
  // ends, collapses any internal run of whitespace to a single space) --
  // so "jose ramirez", "JOSE RAMIREZ", and "Jose   Ramirez" all normalize
  // to the identical search key "jose ramirez", the same key "Jose
  // Ramirez" and the real, accented "José Ramírez" also normalize to.
  function normalizeSearchText(text) {
    return (text || "")
      .normalize("NFD")
      .replace(/[\u0300-\u036f]/g, "")
      .toLowerCase()
      .trim()
      .replace(/\s+/g, " ");
  }

  // Version 1.4.1 Showcase Plays -- an editorial entry point, entirely
  // independent of the player-search/browse section below (no shared
  // state, no shared DOM beyond both living on /explore/). Fetches ONLY
  // showcase.json (a small, fixed-size ~24-row file) once on page load;
  // every value rendered is copied verbatim from that fetched row, never
  // recomputed. A card's "View play" link uses the SAME `/plays/?id=`
  // query-param route the player-browse table uses.
  var SHOWCASE_URL = "showcase.json";

  //: How many cards each group shows before the "Show all" reveal button
  //: appears -- keeps the Showcase scannable/editorial rather than a
  //: second large table dumped above the search bar (see Section 4 of the
  //: v1.4.1 task: "show perhaps first 6 per group prominently").
  var SHOWCASE_INITIAL_VISIBLE = 6;

  function evLaLabel(launchSpeed, launchAngle) {
    var ev =
      launchSpeed === null || launchSpeed === undefined ? "—" : launchSpeed.toFixed(1) + " mph";
    var la =
      launchAngle === null || launchAngle === undefined
        ? "—"
        : Math.round(launchAngle) + "°";
    return ev + " / " + la;
  }

  function buildShowcaseCard(row) {
    var favorable = row.contact_luck_runs >= 0;
    var card = document.createElement("div");
    card.className =
      "showcase-card " + (favorable ? "showcase-card-favorable" : "showcase-card-unfavorable");

    var name = document.createElement("p");
    name.className = "showcase-card-name";
    name.textContent = row.batter_name || "Player " + row.batter_id;
    card.appendChild(name);

    var date = document.createElement("p");
    date.className = "showcase-card-date";
    date.textContent = formatDate(row.game_date);
    card.appendChild(date);

    var rowsEl = document.createElement("div");
    rowsEl.className = "showcase-card-rows";

    [
      ["Recorded result", outcomeLabel(row.outcome_class)],
      ["EV / LA", evLaLabel(row.launch_speed, row.launch_angle)],
      ["Expected RV", formatSigned(row.expected_run_value)],
      ["Final observed RV", formatSigned(row.observed_run_value)],
    ].forEach(function (pair) {
      var rowEl = document.createElement("div");
      rowEl.className = "showcase-card-row";
      var labelEl = document.createElement("span");
      labelEl.className = "showcase-card-row-label";
      labelEl.textContent = pair[0];
      var valueEl = document.createElement("span");
      valueEl.textContent = pair[1];
      rowEl.appendChild(labelEl);
      rowEl.appendChild(valueEl);
      rowsEl.appendChild(rowEl);
    });
    card.appendChild(rowsEl);

    var figure = document.createElement("p");
    figure.className =
      "demo-luck-figure showcase-card-figure " +
      (favorable ? "interval-positive" : "interval-negative");
    figure.textContent = formatSigned(row.contact_luck_runs) + " runs";
    card.appendChild(figure);

    var footer = document.createElement("div");
    footer.className = "showcase-card-footer";
    if (row.interactive_available) {
      var badge = document.createElement("span");
      badge.className = "showcase-card-whatif-badge";
      badge.textContent = "What if? available";
      footer.appendChild(badge);
    }
    var link = document.createElement("a");
    link.className = "showcase-card-link";
    link.href = "/plays/?id=" + encodeURIComponent(row.play_id);
    link.textContent = "View play →";
    footer.appendChild(link);
    card.appendChild(footer);

    return card;
  }

  function renderShowcaseGroup(section, groupKey, rows) {
    var cardsEl = qs('[data-role="showcase-cards-' + groupKey + '"]', section);
    var revealBtn = qs('[data-role="showcase-reveal-' + groupKey + '"]', section);
    if (!cardsEl) return;

    var sorted = rows.slice().sort(function (a, b) {
      return a.rank - b.rank;
    });
    var initiallyVisible = sorted.slice(0, SHOWCASE_INITIAL_VISIBLE);
    var rest = sorted.slice(SHOWCASE_INITIAL_VISIBLE);

    initiallyVisible.forEach(function (row) {
      cardsEl.appendChild(buildShowcaseCard(row));
    });

    if (rest.length && revealBtn) {
      revealBtn.hidden = false;
      revealBtn.textContent = "Show all " + sorted.length;
      revealBtn.addEventListener("click", function () {
        rest.forEach(function (row) {
          cardsEl.appendChild(buildShowcaseCard(row));
        });
        revealBtn.hidden = true;
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
        var favorable = rows.filter(function (r) {
          return r.group === "favorable";
        });
        var unfavorable = rows.filter(function (r) {
          return r.group === "unfavorable";
        });
        renderShowcaseGroup(section, "favorable", favorable);
        renderShowcaseGroup(section, "unfavorable", unfavorable);
        if (loadingEl) loadingEl.hidden = true;
        if (bodyEl) bodyEl.hidden = false;
      })
      .catch(function (err) {
        if (loadingEl) {
          loadingEl.textContent =
            "Showcase Plays could not load right now. The rest of the page is unaffected.";
        }
        if (window.console && window.console.error) window.console.error(err);
      });
  }

  function initExplorePage() {
    var root = qs('[data-role="explore-selected"]');
    if (!root) return;

    var searchInput = qs('[data-role="explore-player-search"]');
    var searchResults = qs('[data-role="explore-player-search-results"]');
    var catalogLoadingEl = qs('[data-role="explore-catalog-loading"]');
    var promptEl = qs('[data-role="explore-prompt"]');
    var selectedNameEl = qs('[data-role="explore-selected-name"]');
    var selectedMetaEl = qs('[data-role="explore-selected-meta"]');
    var changePlayerBtn = qs('[data-role="explore-change-player"]');

    var loadingEl = qs('[data-role="explore-loading"]');
    var emptyEl = qs('[data-role="explore-empty"]');
    var countEl = qs('[data-role="explore-result-count"]');
    var table = qs('[data-role="explore-results-table"]');
    var tbody = qs('[data-role="explore-results-body"]');
    var outcomeSelect = qs('[data-role="explore-outcome-filter"]');
    var luckSelect = qs('[data-role="explore-luck-filter"]');
    var sortSelect = qs('[data-role="explore-sort"]');

    var playersCatalog = [];
    var selectedBatterId = null;
    var selectedRows = [];

    function matchesFilters(row, outcomeFilter, luckFilter) {
      if (outcomeFilter && row.outcome_class !== outcomeFilter) return false;
      if (luckFilter === "favorable" && !(row.contact_luck_runs >= 0)) return false;
      if (luckFilter === "unfavorable" && !(row.contact_luck_runs < 0)) return false;
      return true;
    }

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

    function renderResults() {
      var outcomeFilter = outcomeSelect.value;
      var luckFilter = luckSelect.value;
      var sortKey = sortSelect.value;

      var filtered = selectedRows.filter(function (row) {
        return matchesFilters(row, outcomeFilter, luckFilter);
      });
      var sorter = SORTERS[sortKey] || SORTERS.most_favorable;
      filtered.sort(sorter);

      tbody.innerHTML = "";
      if (!filtered.length) {
        table.hidden = true;
        emptyEl.hidden = false;
        countEl.hidden = true;
        return;
      }
      emptyEl.hidden = true;
      table.hidden = false;
      countEl.hidden = false;
      countEl.textContent = filtered.length + " play" + (filtered.length === 1 ? "" : "s");

      var frag = document.createDocumentFragment();
      filtered.forEach(function (row) {
        var tr = document.createElement("tr");

        var dateTd = document.createElement("td");
        dateTd.textContent = formatDate(row.game_date);
        tr.appendChild(dateTd);

        var batterTd = document.createElement("td");
        var link = document.createElement("a");
        link.className = "player-link";
        // Query-param route (Phase 4.1: a single static /plays/ shell,
        // never one directory per play_id). Absolute path, matching this
        // site's existing routing convention (root_prefix is always "/").
        link.href = "/plays/?id=" + encodeURIComponent(row.play_id);
        link.textContent = row.batter_name || "Player " + row.batter_id;
        batterTd.appendChild(link);
        tr.appendChild(batterTd);

        var evTd = document.createElement("td");
        evTd.className = "numeric";
        evTd.textContent =
          row.launch_speed === null || row.launch_speed === undefined
            ? "—"
            : row.launch_speed.toFixed(1);
        tr.appendChild(evTd);

        var laTd = document.createElement("td");
        laTd.className = "numeric";
        laTd.textContent =
          row.launch_angle === null || row.launch_angle === undefined
            ? "—"
            : Math.round(row.launch_angle) + "°";
        tr.appendChild(laTd);

        var outcomeTd = document.createElement("td");
        outcomeTd.textContent = outcomeLabel(row.outcome_class);
        tr.appendChild(outcomeTd);

        var expectedTd = document.createElement("td");
        expectedTd.className = "numeric";
        expectedTd.textContent = formatSigned(row.expected_run_value);
        tr.appendChild(expectedTd);

        var luckTd = document.createElement("td");
        luckTd.className =
          "numeric explore-luck-col " +
          (row.contact_luck_runs >= 0 ? "interval-positive" : "interval-negative");
        luckTd.textContent = formatSigned(row.contact_luck_runs);
        tr.appendChild(luckTd);

        frag.appendChild(tr);
      });
      tbody.appendChild(frag);
    }

    function showSelectedPlayerShell(entry) {
      promptEl.hidden = true;
      root.hidden = false;
      selectedNameEl.textContent = playerLabel(entry);
      selectedMetaEl.textContent =
        entry.play_count + " scored play" + (entry.play_count === 1 ? "" : "s");
      loadingEl.hidden = false;
      emptyEl.hidden = true;
      countEl.hidden = true;
      table.hidden = true;
      tbody.innerHTML = "";
    }

    function selectPlayer(entry) {
      selectedBatterId = entry.batter_id;
      searchInput.value = playerLabel(entry);
      searchResults.hidden = true;
      showSelectedPlayerShell(entry);

      // The ONE network request this selection makes: exactly this
      // hitter's own play index, never any other batter's file and never
      // the full catalog again.
      fetch("players/" + encodeURIComponent(String(entry.batter_id)) + ".json")
        .then(function (response) {
          if (!response.ok) throw new Error("failed to load player index: " + response.status);
          return response.json();
        })
        .then(function (rows) {
          if (selectedBatterId !== entry.batter_id) return; // superseded by a later selection
          selectedRows = rows;
          loadingEl.hidden = true;
          renderResults();
        })
        .catch(function (err) {
          loadingEl.textContent = "This hitter's plays could not load right now.";
          if (window.console && window.console.error) window.console.error(err);
        });
    }

    function renderSuggestions(matches) {
      searchResults.innerHTML = "";
      if (!matches.length) {
        searchResults.hidden = true;
        return;
      }
      matches.slice(0, MAX_SUGGESTIONS).forEach(function (entry) {
        var item = document.createElement("button");
        item.type = "button";
        item.className = "search-result-item";
        item.textContent =
          playerLabel(entry) + " (" + entry.play_count + " play" + (entry.play_count === 1 ? "" : "s") + ")";
        item.addEventListener("click", function () {
          selectPlayer(entry);
        });
        searchResults.appendChild(item);
      });
      searchResults.hidden = false;
    }

    function onSearchInput() {
      var q = normalizeSearchText(searchInput.value);
      if (!q) {
        renderSuggestions([]);
        return;
      }
      var matches = playersCatalog.filter(function (entry) {
        return normalizeSearchText(entry.batter_name).indexOf(q) !== -1 || String(entry.batter_id) === q;
      });
      renderSuggestions(matches);
    }

    searchInput.addEventListener("input", onSearchInput);
    searchInput.addEventListener("focus", function () {
      if (searchInput.value.trim()) onSearchInput();
    });
    document.addEventListener("click", function (evt) {
      if (!searchResults.contains(evt.target) && evt.target !== searchInput) {
        searchResults.hidden = true;
      }
    });

    changePlayerBtn.addEventListener("click", function () {
      selectedBatterId = null;
      selectedRows = [];
      root.hidden = true;
      promptEl.hidden = false;
      searchInput.value = "";
      searchInput.focus();
    });

    [outcomeSelect, luckSelect, sortSelect].forEach(function (el) {
      el.addEventListener("change", renderResults);
    });

    // The ONLY fetch made on page load -- the small players catalog. No
    // per-player file is ever fetched here or in a loop over
    // playersCatalog; each players/<batter_id>.json fetch happens
    // exclusively inside selectPlayer(), triggered by one user action.
    fetch(PLAYERS_URL)
      .then(function (response) {
        if (!response.ok) throw new Error("failed to load players catalog: " + response.status);
        return response.json();
      })
      .then(function (players) {
        playersCatalog = players;
        catalogLoadingEl.hidden = true;
        promptEl.hidden = false;
        searchInput.disabled = false;

        // Redesign Phase 4: a player page links here with `?batter=<id>`,
        // so the product's stated path -- league, hitter, play -- actually
        // connects instead of dead-ending at an empty picker. This reads
        // the parameter and nothing else; the selection is still not
        // written BACK to the URL, which is Phase 5's job (Explore).
        //
        // An unknown or malformed id is ignored silently and the normal
        // prompt stands: a hitter absent from the published catalog is not
        // an error state, and no extra request is made either way.
        var requested = new URLSearchParams(window.location.search).get("batter");
        if (requested) {
          var match = null;
          for (var i = 0; i < playersCatalog.length; i += 1) {
            if (String(playersCatalog[i].batter_id) === requested) {
              match = playersCatalog[i];
              break;
            }
          }
          if (match) selectPlayer(match);
        }
      })
      .catch(function (err) {
        catalogLoadingEl.textContent = "Players could not load right now. The rest of the site is unaffected.";
        if (window.console && window.console.error) window.console.error(err);
      });
  }

  document.addEventListener("DOMContentLoaded", initShowcaseSection);
  document.addEventListener("DOMContentLoaded", initExplorePage);
})();
