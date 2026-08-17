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
    var sign = value >= 0 ? "+" : "";
    return sign + value.toFixed(2);
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
      var q = searchInput.value.trim().toLowerCase();
      if (!q) {
        renderSuggestions([]);
        return;
      }
      var matches = playersCatalog.filter(function (entry) {
        return (entry.batter_name || "").toLowerCase().indexOf(q) !== -1 || String(entry.batter_id) === q;
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
      })
      .catch(function (err) {
        catalogLoadingEl.textContent = "Players could not load right now. The rest of the site is unaffected.";
        if (window.console && window.console.error) window.console.error(err);
      });
  }

  document.addEventListener("DOMContentLoaded", initExplorePage);
})();
