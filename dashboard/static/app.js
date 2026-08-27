// Contact Luck v1.2 -- client-side leaderboard sort/filter and player search.
// Purely presentational: reads values already rendered into the page
// (row data-* attributes, an embedded player-index JSON script tag) and
// never fetches, recomputes, or requests a score from anywhere.
(function () {
  "use strict";

  // ══ Official ranking tabs ════════════════════════════════════════════════
  // Real `role="tablist"` buttons with a roving tabindex, replacing a pair of
  // `display:none` radios that were absent from the accessibility tree AND
  // the tab order. The two rankings are independent frozen orders; switching
  // between them is navigation, not a filter, and neither is "worst."
  function initRankingTabs() {
    var groups = document.querySelectorAll("[data-role='ranking']");
    Array.prototype.forEach.call(groups, function (group) {
      var tabs = Array.prototype.slice.call(group.querySelectorAll("[role='tab']"));
      if (!tabs.length) return;

      function select(tab, moveFocus) {
        tabs.forEach(function (t) {
          var selected = t === tab;
          t.setAttribute("aria-selected", selected ? "true" : "false");
          t.tabIndex = selected ? 0 : -1;
          var panel = document.getElementById(t.getAttribute("aria-controls"));
          if (!panel) return;
          if (selected) {
            panel.removeAttribute("hidden");
          } else {
            panel.setAttribute("hidden", "");
          }
        });
        if (moveFocus) tab.focus();
      }

      tabs.forEach(function (tab, i) {
        tab.addEventListener("click", function () {
          select(tab, false);
        });
        tab.addEventListener("keydown", function (evt) {
          var next = null;
          if (evt.key === "ArrowRight" || evt.key === "ArrowDown") next = tabs[(i + 1) % tabs.length];
          else if (evt.key === "ArrowLeft" || evt.key === "ArrowUp") next = tabs[(i - 1 + tabs.length) % tabs.length];
          else if (evt.key === "Home") next = tabs[0];
          else if (evt.key === "End") next = tabs[tabs.length - 1];
          if (!next) return;
          evt.preventDefault();
          select(next, true);
        });
      });
    });
  }

  // ══ Accessible sorting ═══════════════════════════════════════════════════
  // The baseline bound a click handler to a bare `<th>`: mouse-only, outside
  // the tab order, no `aria-sort`, no announcement. Here the control is a
  // real button inside the header cell, `aria-sort` tracks the state, and a
  // polite live region says what changed.
  //
  // Rank is a RESTORE, not a two-way toggle. Toggling its direction produced
  // a reverse-rank order that reads like a ranking but is not the official
  // one; there is now exactly one path back to the frozen order and it always
  // means the same thing.
  //
  // Sorting is presentational: it reorders rows already rendered from the
  // snapshot and never recomputes a score, a rank or an interval.
  var SORTED_VIEW_LABEL =
    "Sorted view — not the official Contact Luck ranking. " +
    "The official rank is preserved in its own column.";

  function initLeaderboardSort() {
    var tables = document.querySelectorAll("table.leaderboard[data-sortable]");
    Array.prototype.forEach.call(tables, function (table) {
      var tbody = table.querySelector("tbody");
      var headers = Array.prototype.slice.call(table.querySelectorAll("th[data-sort-key]"));
      var label = document.querySelector(table.getAttribute("data-sorted-label-target"));
      var status = document.querySelector(table.getAttribute("data-sort-status-target"));

      function apply(th, dir) {
        var key = th.getAttribute("data-sort-key");
        var type = th.getAttribute("data-sort-type") || "number";
        var restores = th.hasAttribute("data-sort-restores");

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

        headers.forEach(function (h) {
          h.setAttribute("aria-sort", "none");
        });
        th.setAttribute("aria-sort", dir === "asc" ? "ascending" : "descending");

        // The three simultaneous expressions of sorted state
        // (docs/design/tables.md rule 5): `aria-sort` above, the visible
        // label, and the official accent leaving the rank column -- the last
        // driven by this one attribute, which the stylesheet keys off.
        if (restores) {
          table.removeAttribute("data-sorted");
          if (label) label.textContent = "";
          if (status) status.textContent = "Official ranking restored, by rank ascending.";
        } else {
          table.setAttribute("data-sorted", "");
          if (label) label.textContent = SORTED_VIEW_LABEL;
          if (status) {
            status.textContent =
              "Sorted by " +
              th.textContent.trim().split("\n")[0] +
              ", " +
              (dir === "asc" ? "ascending" : "descending") +
              ". This is a sorted view, not the official ranking.";
          }
        }
      }

      headers.forEach(function (th) {
        var button = th.querySelector("[data-role='sort']");
        if (!button) return;
        button.addEventListener("click", function () {
          var restores = th.hasAttribute("data-sort-restores");
          // The restore control has one direction, always. Everything else
          // toggles, defaulting to descending -- "most of this first" is what
          // a reader wants from a leaderboard column.
          var dir;
          if (restores) {
            dir = "asc";
          } else {
            dir = th.getAttribute("aria-sort") === "descending" ? "asc" : "desc";
          }
          apply(th, dir);
        });
      });
    });
  }

  // ══ Filter ═══════════════════════════════════════════════════════════════
  // Same matching behaviour as the baseline, now with a real label, a result
  // count in a polite live region, and an empty state that says what to do.
  function initLeaderboardFilter() {
    var inputs = document.querySelectorAll("input[data-role='leaderboard-filter']");
    Array.prototype.forEach.call(inputs, function (input) {
      var targetSelector = input.getAttribute("data-target-table");
      var table = targetSelector ? document.querySelector(targetSelector) : null;
      if (!table) return;
      var tbody = table.querySelector("tbody");
      var status = document.querySelector(input.getAttribute("data-status-target"));
      var empty = document.querySelector(
        "[data-role='filter-empty'][data-for-table='" + targetSelector + "']"
      );

      input.addEventListener("input", function () {
        var q = normalizeSearchText(input.value);
        var shown = 0;
        Array.prototype.forEach.call(tbody.querySelectorAll("tr"), function (row) {
          var name = normalizeSearchText(row.dataset.playerName || "");
          var match = !q || name.indexOf(q) !== -1;
          row.hidden = !match;
          if (match) shown += 1;
        });
        if (empty) {
          if (shown === 0) {
            empty.removeAttribute("hidden");
          } else {
            empty.setAttribute("hidden", "");
          }
        }
        if (status) {
          status.textContent = !q
            ? ""
            : shown === 0
              ? "No hitters match."
              : shown === 1
                ? "1 hitter matches."
                : shown + " hitters match.";
        }
      });
    });
  }

  // Search matching only -- NEVER used for display (render() below always
  // renders the original, accented p.batter_name verbatim). Identical
  // semantics to dashboard/static/explore.js's normalizeSearchText (kept as
  // a duplicated pure function rather than a shared import -- app.js and
  // explore.js are independent, unbundled <script> tags, matching this
  // repo's existing convention of duplicating small pure functions across
  // read-only-boundary modules, e.g. dashboard/snapshot_data.py's
  // parse_snapshot_directory_name): Unicode NFD-decomposes an accented
  // character into a base letter plus a combining mark (e.g. U+00E9
  // "e-acute" -> "e" + U+0301); \u0300-\u036f is the Unicode "Combining
  // Diacritical Marks" block, stripped here so only the base letter remains, then
  // lowercased and whitespace-normalized (trimmed, internal runs collapsed
  // to one space) -- so "jose ramirez", "JOSE RAMIREZ", "Jose   Ramirez",
  // and the real "José Ramírez" all normalize to the same search key.
  function normalizeSearchText(text) {
    return (text || "")
      .normalize("NFD")
      .replace(/[\u0300-\u036f]/g, "")
      .toLowerCase()
      .trim()
      .replace(/\s+/g, " ");
  }

  // ══ Mobile navigation disclosure ═════════════════════════════════════════
  // Redesign Phase 2. The routes are a plain <ul> that CSS shows inline from
  // 640px up; below that the list is collapsed behind a named "Menu" control.
  // The list ships with the `hidden` attribute so a JS-less mobile browser is
  // not stuck with an open menu, and base.html's <noscript> block restores it
  // there. Everything here is presentation only.
  function initSiteNavDisclosure() {
    var toggle = document.querySelector("[data-role='site-nav-toggle']");
    var list = document.querySelector("[data-role='site-nav-list']");
    if (!toggle || !list) return;

    function setOpen(open) {
      toggle.setAttribute("aria-expanded", open ? "true" : "false");
      if (open) {
        list.removeAttribute("hidden");
      } else {
        list.setAttribute("hidden", "");
      }
    }

    toggle.addEventListener("click", function () {
      setOpen(toggle.getAttribute("aria-expanded") !== "true");
    });

    document.addEventListener("keydown", function (evt) {
      if (evt.key !== "Escape") return;
      if (toggle.getAttribute("aria-expanded") !== "true") return;
      setOpen(false);
      toggle.focus();
    });

    document.addEventListener("click", function (evt) {
      if (toggle.getAttribute("aria-expanded") !== "true") return;
      if (toggle.contains(evt.target) || list.contains(evt.target)) return;
      setOpen(false);
    });
  }

  // ══ Global player search · ARIA 1.2 combobox with listbox popup ══════════
  // Redesign Phase 2. The baseline was an unlabelled <input> writing <a>
  // elements into a <div>: no combobox role, no aria-expanded, no keyboard
  // navigation, and no Escape. This is the one implementation of the pattern;
  // Explore's hitter picker adopts it in Phase 5.
  //
  // Focus never leaves the input -- that is the combobox contract. The active
  // option is pointed at with `aria-activedescendant`, not with DOM focus.
  //
  // Options are <li role="option"> wrapping a real <a href>. `option` is a
  // children-presentational role, so assistive technology reads the option's
  // own accessible name and never sees a nested link; a mouse user still gets
  // real link behaviour (middle-click, open in new tab, status-bar target).
  // Enter is handled here because the anchor is not focusable.
  //
  // NOTHING is computed here. The index is a build-time JSON blob of
  // id/name/url plus a `ranked` boolean copied from the snapshot's own
  // qualification status (dashboard/build.py) -- no score, rank, interval or
  // probability is derived in this file.
  var SEARCH_RESULT_LIMIT = 8;

  function initGlobalPlayerSearch() {
    var wrap = document.querySelector("[data-role='global-search']");
    var input = document.querySelector("[data-role='global-player-search']");
    var listbox = document.querySelector("[data-role='global-player-search-results']");
    var statusEl = document.querySelector("[data-role='global-player-search-status']");
    var closeBtn = document.querySelector("[data-role='global-search-close']");
    var dataEl = document.getElementById("player-index-data");
    if (!wrap || !input || !listbox || !dataEl) return;

    var players;
    try {
      players = JSON.parse(dataEl.textContent);
    } catch (err) {
      return;
    }

    var options = [];
    var activeIndex = -1;
    // The mobile full-screen sheet is a structural transformation, not a
    // narrower dropdown -- it applies only below the mobile breakpoint, which
    // is the same 640px boundary the stylesheet uses.
    var sheetQuery = window.matchMedia("(max-width: 639px)");

    function announce(text) {
      if (statusEl) statusEl.textContent = text;
    }

    function setExpanded(expanded) {
      input.setAttribute("aria-expanded", expanded ? "true" : "false");
      if (expanded) {
        listbox.removeAttribute("hidden");
      } else {
        listbox.setAttribute("hidden", "");
      }
    }

    function setActive(index) {
      if (activeIndex >= 0 && options[activeIndex]) {
        options[activeIndex].setAttribute("aria-selected", "false");
      }
      activeIndex = index;
      if (activeIndex >= 0 && options[activeIndex]) {
        var el = options[activeIndex];
        el.setAttribute("aria-selected", "true");
        input.setAttribute("aria-activedescendant", el.id);
        if (el.scrollIntoView) el.scrollIntoView({ block: "nearest" });
      } else {
        input.removeAttribute("aria-activedescendant");
      }
    }

    function openSheet() {
      if (!sheetQuery.matches) return;
      document.body.classList.add("search-sheet-open");
      if (closeBtn) closeBtn.removeAttribute("hidden");
    }

    function closeSheet() {
      document.body.classList.remove("search-sheet-open");
      if (closeBtn) closeBtn.setAttribute("hidden", "");
    }

    function close() {
      setExpanded(false);
      setActive(-1);
      closeSheet();
    }

    function render(matches, query) {
      listbox.innerHTML = "";
      options = [];
      setActive(-1);

      if (!query) {
        setExpanded(false);
        announce("");
        return;
      }

      if (!matches.length) {
        var empty = document.createElement("li");
        // Not an option: an empty state must not be reachable by Up/Down or
        // selectable by Enter. `presentation` keeps it out of the listbox's
        // option set while leaving the text visible and announced by the
        // status region below.
        empty.setAttribute("role", "presentation");
        empty.className = "global-search-empty";
        empty.textContent = "No player matches “" + query + "”. Try a surname, or an MLBAM ID.";
        listbox.appendChild(empty);
        setExpanded(true);
        announce("No players found.");
        return;
      }

      matches.slice(0, SEARCH_RESULT_LIMIT).forEach(function (p, i) {
        var li = document.createElement("li");
        li.id = "global-player-search-option-" + i;
        li.className = "global-search-option";
        li.setAttribute("role", "option");
        li.setAttribute("aria-selected", "false");
        li.dataset.url = p.url;

        var name = p.batter_name || "Player " + p.batter_id;

        var a = document.createElement("a");
        a.href = p.url;
        a.tabIndex = -1;
        a.textContent = name;
        li.appendChild(a);

        // Unqualified players are searchable, findable and MARKED -- never
        // suppressed, never de-emphasised (a preserved product invariant).
        // The wording matches the player page's own "No official rank".
        if (p.ranked === false) {
          var note = document.createElement("span");
          note.className = "global-search-note";
          note.textContent = "No official rank";
          li.appendChild(note);
        }

        // `option` is a children-presentational role, so without this the
        // accessible name is the concatenated text content and a screen
        // reader announces "Jose AltuveNo official rank" as one run-on word.
        li.setAttribute(
          "aria-label",
          p.ranked === false ? name + ", no official rank" : name
        );

        li.addEventListener("mousedown", function (evt) {
          evt.preventDefault();
          window.location.assign(p.url);
        });

        listbox.appendChild(li);
        options.push(li);
      });

      setExpanded(true);
      var shown = Math.min(matches.length, SEARCH_RESULT_LIMIT);
      announce(
        shown === 1
          ? "1 player found."
          : shown + " players found" + (matches.length > shown ? ", showing the first " + shown : "") + "."
      );
    }

    function search() {
      var q = normalizeSearchText(input.value);
      if (!q) {
        render([], "");
        return;
      }
      var matches = players.filter(function (p) {
        return normalizeSearchText(p.batter_name).indexOf(q) !== -1 || String(p.batter_id) === q;
      });
      render(matches, input.value.trim());
    }

    input.addEventListener("input", search);

    input.addEventListener("focus", function () {
      openSheet();
      if (input.value.trim()) search();
    });

    input.addEventListener("keydown", function (evt) {
      var isOpen = input.getAttribute("aria-expanded") === "true";

      if (evt.key === "ArrowDown" || evt.key === "ArrowUp") {
        if (!isOpen) {
          search();
          return;
        }
        if (!options.length) return;
        evt.preventDefault();
        var next;
        if (evt.key === "ArrowDown") {
          next = activeIndex + 1 >= options.length ? 0 : activeIndex + 1;
        } else {
          next = activeIndex - 1 < 0 ? options.length - 1 : activeIndex - 1;
        }
        setActive(next);
        return;
      }

      if (evt.key === "Home" && isOpen && options.length) {
        evt.preventDefault();
        setActive(0);
        return;
      }
      if (evt.key === "End" && isOpen && options.length) {
        evt.preventDefault();
        setActive(options.length - 1);
        return;
      }

      if (evt.key === "Enter") {
        if (!isOpen || !options.length) return;
        evt.preventDefault();
        var target = options[activeIndex >= 0 ? activeIndex : 0];
        if (target && target.dataset.url) window.location.assign(target.dataset.url);
        return;
      }

      if (evt.key === "Escape") {
        // First Escape dismisses the list; a second clears the query. Both
        // keep focus in the field, which is where the reader is.
        evt.preventDefault();
        if (isOpen) {
          close();
        } else if (input.value) {
          input.value = "";
          announce("");
          closeSheet();
        } else {
          closeSheet();
        }
        return;
      }

      if (evt.key === "Tab") {
        close();
      }
    });

    if (closeBtn) {
      closeBtn.addEventListener("click", function () {
        close();
        input.blur();
      });
    }

    document.addEventListener("click", function (evt) {
      if (wrap.contains(evt.target)) return;
      close();
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    initRankingTabs();
    initLeaderboardSort();
    initLeaderboardFilter();
    initSiteNavDisclosure();
    initGlobalPlayerSearch();
  });
})();
