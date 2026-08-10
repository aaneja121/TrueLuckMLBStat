// Contact Luck v1.2 -- client-side leaderboard sort/filter and player search.
// Purely presentational: reads values already rendered into the page
// (row data-* attributes, an embedded player-index JSON script tag) and
// never fetches, recomputes, or requests a score from anywhere.
(function () {
  "use strict";

  function initLeaderboardSort() {
    var tables = document.querySelectorAll("table.leaderboard[data-sortable]");
    Array.prototype.forEach.call(tables, function (table) {
      var tbody = table.querySelector("tbody");
      var labelSelector = table.getAttribute("data-sorted-label-target");
      var label = labelSelector ? document.querySelector(labelSelector) : null;
      var headers = table.querySelectorAll("th[data-sort-key]");
      var currentSort = null;

      Array.prototype.forEach.call(headers, function (th) {
        th.addEventListener("click", function () {
          var key = th.getAttribute("data-sort-key");
          var type = th.getAttribute("data-sort-type") || "number";
          var dir = currentSort && currentSort.key === key && currentSort.dir === "desc" ? "asc" : "desc";
          currentSort = { key: key, dir: dir };

          var rows = Array.prototype.slice.call(tbody.querySelectorAll("tr"));
          rows.sort(function (a, b) {
            var av = a.dataset[key];
            var bv = b.dataset[key];
            if (type === "number") {
              av = parseFloat(av);
              bv = parseFloat(bv);
            }
            if (av < bv) return dir === "asc" ? -1 : 1;
            if (av > bv) return dir === "asc" ? 1 : -1;
            return 0;
          });
          rows.forEach(function (row) {
            tbody.appendChild(row);
          });

          if (label) {
            label.textContent =
              key === "officialRank"
                ? ""
                : "Sorted view — not the official Contact Luck ranking. " +
                  "The official rank is preserved in its own column.";
          }
        });
      });
    });
  }

  function initLeaderboardFilter() {
    var inputs = document.querySelectorAll("input[data-role='leaderboard-filter']");
    Array.prototype.forEach.call(inputs, function (input) {
      var targetSelector = input.getAttribute("data-target-table");
      var table = targetSelector ? document.querySelector(targetSelector) : null;
      if (!table) return;
      var tbody = table.querySelector("tbody");
      input.addEventListener("input", function () {
        var q = input.value.trim().toLowerCase();
        Array.prototype.forEach.call(tbody.querySelectorAll("tr"), function (row) {
          var name = (row.dataset.playerName || "").toLowerCase();
          row.style.display = !q || name.indexOf(q) !== -1 ? "" : "none";
        });
      });
    });
  }

  function initGlobalPlayerSearch() {
    var input = document.querySelector("[data-role='global-player-search']");
    var resultsBox = document.querySelector("[data-role='global-player-search-results']");
    var dataEl = document.getElementById("player-index-data");
    if (!input || !resultsBox || !dataEl) return;

    var players;
    try {
      players = JSON.parse(dataEl.textContent);
    } catch (err) {
      return;
    }

    function render(matches) {
      resultsBox.innerHTML = "";
      if (!matches.length) {
        resultsBox.hidden = true;
        return;
      }
      matches.slice(0, 8).forEach(function (p) {
        var a = document.createElement("a");
        a.href = p.url;
        a.className = "search-result-item";
        a.textContent = p.batter_name || "Player " + p.batter_id;
        resultsBox.appendChild(a);
      });
      resultsBox.hidden = false;
    }

    input.addEventListener("input", function () {
      var q = input.value.trim().toLowerCase();
      if (!q) {
        render([]);
        return;
      }
      var matches = players.filter(function (p) {
        return (p.batter_name || "").toLowerCase().indexOf(q) !== -1 || String(p.batter_id) === q;
      });
      render(matches);
    });

    input.addEventListener("focus", function () {
      if (input.value.trim()) input.dispatchEvent(new Event("input"));
    });

    document.addEventListener("click", function (evt) {
      if (!resultsBox.contains(evt.target) && evt.target !== input) {
        resultsBox.hidden = true;
      }
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    initLeaderboardSort();
    initLeaderboardFilter();
    initGlobalPlayerSearch();
  });
})();
