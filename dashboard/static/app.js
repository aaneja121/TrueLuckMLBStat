// Contact Luck v1.2 -- client-side leaderboard sort/filter and player search.
// Purely presentational: reads values already rendered into the page
// (row data-* attributes, an embedded player-index JSON script tag) and
// never fetches, recomputes, or requests a score from anywhere.
(function () {
  "use strict";

  function prefersReducedMotion() {
    return !!(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches);
  }

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

          // FLIP re-rank animation: record every row's pre-sort viewport
          // position (keyed by the row element itself, since `rows` gets
          // reordered in place by the sort below -- an index-based map
          // would silently pair the wrong before/after position). Motion
          // only; the sort result itself is identical either way.
          var reduceMotion = prefersReducedMotion();
          var firstPositions = null;
          if (!reduceMotion) {
            firstPositions = new Map();
            rows.forEach(function (row) {
              firstPositions.set(row, row.getBoundingClientRect().top);
            });
          }

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

          if (firstPositions) {
            table.classList.add("leaderboard-animating");
            rows.forEach(function (row) {
              var delta = firstPositions.get(row) - row.getBoundingClientRect().top;
              if (delta) row.style.transform = "translateY(" + delta + "px)";
            });
            // Force a reflow so the browser registers the transform above
            // as a real starting frame before it gets cleared below --
            // without this the two style writes coalesce and nothing
            // animates.
            void tbody.offsetHeight;
            rows.forEach(function (row) {
              row.style.transform = "";
            });
            window.setTimeout(function () {
              table.classList.remove("leaderboard-animating");
            }, 400);
          }

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

  // Count-up animation for hero-scale stat readouts
  // (`[data-count-target]` -- homepage stat ticker, player hero figure).
  // The server-rendered textContent is ALWAYS already the correct final
  // value (see index.html/player.html) -- this only replaces it
  // temporarily during the animation and restores the exact same
  // formatted value at the end, so a no-JS or reduced-motion visitor sees
  // the identical, always-correct number with no animation at all.
  function initCountUp() {
    var targets = document.querySelectorAll("[data-count-target]");
    if (!targets.length || prefersReducedMotion() || !("IntersectionObserver" in window)) return;

    function formatValue(el, value) {
      var decimals = parseInt(el.getAttribute("data-count-decimals") || "0", 10);
      var signed = el.getAttribute("data-count-signed") === "true";
      var suffix = el.getAttribute("data-count-suffix") || "";
      var out = value.toFixed(decimals);
      if (signed && value >= 0) out = "+" + out;
      return out + suffix;
    }

    function animate(el) {
      var to = parseFloat(el.getAttribute("data-count-to"));
      if (isNaN(to)) return;
      var duration = 900;
      var start = null;
      function step(ts) {
        if (start === null) start = ts;
        var progress = Math.min(1, (ts - start) / duration);
        var eased = 1 - Math.pow(1 - progress, 3);
        el.textContent = formatValue(el, to * eased);
        if (progress < 1) {
          window.requestAnimationFrame(step);
        } else {
          el.textContent = formatValue(el, to);
        }
      }
      window.requestAnimationFrame(step);
    }

    var observer = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            animate(entry.target);
            observer.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.4 }
    );
    Array.prototype.forEach.call(targets, function (el) {
      observer.observe(el);
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
      var q = normalizeSearchText(input.value);
      if (!q) {
        render([]);
        return;
      }
      var matches = players.filter(function (p) {
        return normalizeSearchText(p.batter_name).indexOf(q) !== -1 || String(p.batter_id) === q;
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

  //: Minimum horizontal gap (px) enforced between two adjacent Luck
  //: Spectrum labels' rendered edges -- keep in sync with the identical
  //: copy of this constant/function in static/explore.js (independent,
  //: unbundled <script> tags -- see that file's own comment on why this
  //: is duplicated rather than shared).
  var SPECTRUM_LABEL_GAP_PX = 12;

  // Two markers close in Contact Luck VALUE land close in PIXEL position
  // too, often close enough that their labels would overlap even though
  // each is individually centered on its own dot -- unresolvable at
  // server-render time (dashboard/build.py), since it depends on each
  // label's actual rendered width and the track's actual pixel width.
  //
  // Two earlier versions of this got progressively closer but each had a
  // real bug, both caught by an actual pixel-geometry check (not just a
  // glance at a screenshot):
  //   1. Walking labels left to right, pushing each colliding one just
  //      far enough past the previous one, compounds for a cluster of 3+:
  //      the last label in a tight cluster could end up 150-200px from
  //      its own dot (real user feedback: "these names are floating off
  //      to the side").
  //   2. Grouping mutually-overlapping markers and centering each group
  //      independently around its own natural midpoint fixed that, but a
  //      group's centered width can EXPAND past where a neighboring,
  //      previously-untouched single marker naturally sat, creating a
  //      brand new overlap between two markers that were never in the
  //      same group.
  // This version does both, in order: (a) group + center each cluster
  // around its own midpoint (identical to version 2 -- still the
  // minimum-shift choice for markers that are actually close), THEN
  // (b) treat each cluster as a rigid block and push blocks apart, left
  // to right, exactly like version 1's cascade -- but operating on whole
  // (already-internally-resolved) blocks instead of individual labels, so
  // the cascade only ever has to close a genuinely leftover gap between
  // two groups, never re-spread an already-centered group.
  function declutterLuckSpectrum(trackEl) {
    // Below the mobile breakpoint the track becomes a wrapping chip list
    // (position: static, see style.css) -- applying a pixel margin-left
    // nudge there would corrupt that flex layout instead of fixing
    // anything, since there's no absolute-positioned label to "collide"
    // in the first place.
    if (window.matchMedia && window.matchMedia("(max-width: 640px)").matches) return;
    // Unlabeled-tier markers keep their `.luck-spectrum-marker-text` in the
    // DOM (for the accessibility tree/hover-reveal, see style.css) but it
    // has no persistent visible position to defend -- including it here
    // would measure a hidden (opacity:0) element's real layout box and
    // could nudge a genuinely visible label to avoid "colliding" with text
    // nobody sees.
    var markers = Array.prototype.slice.call(
      trackEl.querySelectorAll(".luck-spectrum-marker:not(.luck-spectrum-marker-tier-unlabeled)")
    );
    if (markers.length < 2) return;

    var items = markers
      .map(function (m) {
        var textEl = m.querySelector(".luck-spectrum-marker-text");
        var r = textEl.getBoundingClientRect();
        return { marker: m, textEl: textEl, width: r.width, centerX: r.left + r.width / 2 };
      })
      .sort(function (a, b) {
        return a.centerX - b.centerX;
      });

    // Phase A: group into clusters using each item's NATURAL (unshifted)
    // extent -- two adjacent items are in the same cluster if their true
    // bounding boxes would overlap (with the same minimum gap declutter
    // enforces) -- then compute each cluster's ideal, internally-centered
    // layout as a block. Nothing is applied to the DOM yet.
    var clusters = [[items[0]]];
    for (var i = 1; i < items.length; i++) {
      var prevItem = items[i - 1];
      var cur = items[i];
      var prevNaturalRight = prevItem.centerX + prevItem.width / 2;
      var curNaturalLeft = cur.centerX - cur.width / 2;
      if (curNaturalLeft < prevNaturalRight + SPECTRUM_LABEL_GAP_PX) {
        clusters[clusters.length - 1].push(cur);
      } else {
        clusters.push([cur]);
      }
    }

    var blocks = clusters.map(function (cluster) {
      var totalWidth =
        cluster.reduce(function (sum, it) {
          return sum + it.width;
        }, 0) +
        SPECTRUM_LABEL_GAP_PX * (cluster.length - 1);
      var naturalCenter = (cluster[0].centerX + cluster[cluster.length - 1].centerX) / 2;
      var start = naturalCenter - totalWidth / 2;
      var x = start;
      cluster.forEach(function (it) {
        it.targetCenter = x + it.width / 2;
        x += it.width + SPECTRUM_LABEL_GAP_PX;
      });
      return { items: cluster, start: start, end: start + totalWidth };
    });

    // Phase B: push whole BLOCKS apart (never re-spreading a block's own
    // already-centered internal layout) so two different clusters' label
    // groups can never overlap each other either.
    for (var b = 1; b < blocks.length; b++) {
      var prevBlock = blocks[b - 1];
      var curBlock = blocks[b];
      if (curBlock.start < prevBlock.end + SPECTRUM_LABEL_GAP_PX) {
        var delta = prevBlock.end + SPECTRUM_LABEL_GAP_PX - curBlock.start;
        curBlock.start += delta;
        curBlock.end += delta;
        curBlock.items.forEach(function (it) {
          it.targetCenter += delta;
        });
      }
    }

    // Apply: every item now has a final targetCenter (identical to its
    // own true centerX, i.e. shift 0, unless it needed to move as part of
    // resolving its own cluster or a later block push).
    items.forEach(function (it) {
      var shift = it.targetCenter - it.centerX;
      if (Math.abs(shift) <= 0.5) return;
      it.textEl.style.marginLeft = shift.toFixed(1) + "px";
      // The dot-anchor tick (style.css's `.luck-spectrum-marker::before`)
      // is only 9px tall -- fine for an unshifted label directly beneath
      // it, but not enough to visually tie a label to its dot once it's
      // been nudged sideways. This horizontal leader continues that tick
      // across to wherever the label ended up (in either direction), so
      // the connection reads as one bent line instead of something
      // inferred from proximity.
      var leader = document.createElement("span");
      leader.className = "luck-spectrum-marker-leader";
      leader.setAttribute("aria-hidden", "true");
      if (shift >= 0) {
        leader.style.left = "0";
        leader.style.width = shift.toFixed(1) + "px";
      } else {
        leader.style.left = shift.toFixed(1) + "px";
        leader.style.width = (-shift).toFixed(1) + "px";
      }
      it.marker.appendChild(leader);
    });
  }

  function initLuckSpectrums() {
    var tracks = document.querySelectorAll(".luck-spectrum-track");
    Array.prototype.forEach.call(tracks, declutterLuckSpectrum);
  }

  //: Mobile header nav (base.html's `.nav-toggle` button + `#site-nav-panel`
  //: wrapper around the nav links/search/data-through badge). Progressive
  //: enhancement, not a hard JS dependency: the button ships `hidden` and
  //: the panel ships fully visible in the HTML, so a page with JS disabled
  //: (or still loading) shows mobile nav exactly as it always has --
  //: stacked, always-open. Only once this runs does the button appear and
  //: the panel collapse behind it. At desktop widths (see style.css)
  //: `#site-nav-panel` is `display: contents` regardless of any class this
  //: adds, so none of this has any visual effect above the 640px
  //: breakpoint -- the collapse/expand classes simply do nothing there.
  function initNavToggle() {
    var toggle = document.querySelector(".nav-toggle");
    var panel = document.getElementById("site-nav-panel");
    if (!toggle || !panel) return;

    function setOpen(open) {
      panel.classList.toggle("nav-collapsed", !open);
      toggle.setAttribute("aria-expanded", open ? "true" : "false");
    }

    toggle.hidden = false;
    setOpen(false);

    toggle.addEventListener("click", function () {
      setOpen(panel.classList.contains("nav-collapsed"));
    });

    document.addEventListener("click", function (evt) {
      if (
        !panel.classList.contains("nav-collapsed") &&
        !panel.contains(evt.target) &&
        evt.target !== toggle &&
        !toggle.contains(evt.target)
      ) {
        setOpen(false);
      }
    });

    document.addEventListener("keydown", function (evt) {
      if (evt.key === "Escape" && !panel.classList.contains("nav-collapsed")) {
        setOpen(false);
        toggle.focus();
      }
    });

    // A resize/orientation change back past the header's own collapse
    // breakpoint (see style.css's `@media (max-width: 1279px)` -- wider
    // than the site's usual 640px mobile cutoff, kept in sync with this
    // value) must never leave the panel stuck collapsed once CSS switches
    // it back to the always-visible desktop layout (`display: contents`)
    // -- the class itself is harmless there, but this keeps `aria-expanded`
    // honest too.
    var mq = window.matchMedia("(max-width: 1279px)");
    function handleBreakpointChange() {
      if (!mq.matches) setOpen(true);
    }
    if (mq.addEventListener) mq.addEventListener("change", handleBreakpointChange);
    else if (mq.addListener) mq.addListener(handleBreakpointChange);
  }

  document.addEventListener("DOMContentLoaded", function () {
    initLeaderboardSort();
    initLeaderboardFilter();
    initGlobalPlayerSearch();
    initCountUp();
    initLuckSpectrums();
    initNavToggle();
  });
})();
