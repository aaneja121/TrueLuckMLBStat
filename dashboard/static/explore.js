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

  // Real MLB headshot avatar, matching dashboard/templates/_macros.html's
  // `player_avatar` markup/behavior exactly (same public img.mlbstatic.com
  // URL pattern, same fallback-underneath-image DOM order, same inline
  // `onerror` -- duplicated here rather than shared, matching this file's
  // existing convention of duplicating small pure functions across
  // independent, unbundled <script> tags, e.g. normalizeSearchText above).
  function buildAvatar(batterId, sizeClass) {
    var wrap = document.createElement("span");
    wrap.className = "player-avatar" + (sizeClass ? " " + sizeClass : "");

    var fallback = document.createElement("span");
    fallback.className = "player-avatar-fallback";
    fallback.setAttribute("aria-hidden", "true");
    fallback.innerHTML =
      '<svg viewBox="0 0 24 24" focusable="false">' +
      '<circle cx="12" cy="8.5" r="4" fill="none" stroke="currentColor" stroke-width="1.6"></circle>' +
      '<path d="M4 20c1.4-4.2 4.4-6.2 8-6.2s6.6 2 8 6.2" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"></path>' +
      "</svg>";
    wrap.appendChild(fallback);

    var img = document.createElement("img");
    img.src =
      "https://img.mlbstatic.com/mlb-photos/image/upload/w_180,q_auto:best/v1/people/" +
      encodeURIComponent(String(batterId)) +
      "/headshot/67/current";
    img.alt = "";
    img.loading = "lazy";
    img.decoding = "async";
    img.onerror = function () {
      img.style.display = "none";
    };
    wrap.appendChild(img);

    return wrap;
  }

  function evLaLabel(launchSpeed, launchAngle) {
    var ev =
      launchSpeed === null || launchSpeed === undefined ? "—" : launchSpeed.toFixed(1) + " mph";
    var la =
      launchAngle === null || launchAngle === undefined
        ? "—"
        : Math.round(launchAngle) + "°";
    return ev + " / " + la;
  }

  function buildShowcaseCard(row, featured) {
    var favorable = row.contact_luck_runs >= 0;
    var card = document.createElement("div");
    card.className =
      "showcase-card " +
      (favorable ? "showcase-card-favorable" : "showcase-card-unfavorable") +
      (featured ? " showcase-card-featured" : "");

    var header = document.createElement("div");
    header.className = "showcase-card-header";
    header.appendChild(buildAvatar(row.batter_id, featured ? "player-avatar-lg" : "player-avatar-sm"));

    var name = document.createElement("p");
    name.className = "showcase-card-name";
    name.textContent = row.batter_name || "Player " + row.batter_id;
    header.appendChild(name);
    card.appendChild(header);

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

    // The single most extreme play in the group (index 0 of the rank-
    // sorted list) is rendered as the featured card -- see
    // `.showcase-card-featured` in style.css. Only ever the FIRST card of
    // the initially-visible set, never a "Show all"-revealed one.
    initiallyVisible.forEach(function (row, index) {
      cardsEl.appendChild(buildShowcaseCard(row, index === 0));
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

  // ---- Luck Spectrum (client-rendered twin of dashboard/templates/
  // _macros.html::luck_spectrum) -- see that macro's own docstring for the
  // shared contract. `spectrumPositionPct`/the edge logic below are a
  // deliberate line-for-line port of `visuals.spectrum_position_pct` /
  // `build.py`'s `_spectrum`, so a value plotted here lands at the
  // identical position/label placement a server-rendered spectrum would
  // give it. Keep the two in sync if either ever changes.
  var SPECTRUM_EDGE_MARGIN_PCT = 10;

  //: Minimum horizontal gap (px) `declutterLuckSpectrum` (duplicated in
  //: static/app.js -- see that copy's own comment on why) enforces
  //: between two adjacent labels' rendered edges.
  var SPECTRUM_LABEL_GAP_PX = 12;

  // Two markers close in VALUE land close in PIXEL position too, often
  // close enough that their labels would overlap even though each is
  // individually centered on its own dot. This can only be resolved at
  // RUNTIME, in the browser, because it depends on each label's actual
  // rendered width (font metrics, player name length) and the track's
  // actual pixel width -- neither exists at build time.
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
  // two groups, never re-spread an already-centered group. Called once,
  // after the spectrum is in the document (so getBoundingClientRect
  // reflects real layout).
  function declutterLuckSpectrum(trackEl) {
    if (!trackEl) return;
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

  function computeSpectrumDomain(values) {
    var all = values.concat([0]);
    var lo = Math.min.apply(null, all);
    var hi = Math.max.apply(null, all);
    var span = hi - lo || 1;
    var pad = span * 0.12;
    return [lo - pad, hi + pad];
  }

  function spectrumPositionPct(value, domain) {
    var span = domain[1] - domain[0];
    if (span <= 0) return 50;
    var pct = ((value - domain[0]) / span) * 100;
    return Math.max(0, Math.min(100, pct));
  }

  //: Minimum separation between two labeled points on the SAME side of
  //: zero, as a fraction of that side's own half-range (the most extreme
  //: value's own magnitude) -- a deliberate line-for-line port of
  //: `dashboard/spectrum_labels.py`'s `DEFAULT_MIN_SEPARATION_FRACTION`.
  //: Keep the two in sync if either ever changes -- see that module's own
  //: docstring for the full rationale and the edge cases this covers.
  var SPECTRUM_MIN_SEPARATION_FRACTION = 0.12;

  // Line-for-line port of spectrum_labels.select_spectrum_labels (Python).
  // `values` must all be on the same side of zero. Returns an array
  // parallel to `values`: each entry is that point's label rank (0 =
  // primary/most prominent, 1 = secondary, ...) if it should be labeled,
  // or null if it should render as an unlabeled dot only.
  function selectSpectrumLabels(values, maxLabels, minSeparationFraction) {
    var n = values.length;
    var result = new Array(n).fill(null);
    if (n === 0 || maxLabels <= 0) return result;

    var order = values
      .map(function (v, i) {
        return i;
      })
      .sort(function (a, b) {
        return Math.abs(values[b]) - Math.abs(values[a]);
      });
    var halfRange = Math.abs(values[order[0]]);
    if (halfRange <= 0) {
      result[order[0]] = 0;
      return result;
    }

    var threshold = halfRange * minSeparationFraction;
    var chosen = [];
    for (var idx = 0; idx < order.length; idx++) {
      if (chosen.length >= maxLabels) break;
      var i = order[idx];
      var qualifies = chosen.every(function (j) {
        return Math.abs(values[i] - values[j]) >= threshold;
      });
      if (qualifies) chosen.push(i);
    }
    chosen.forEach(function (i, rank) {
      result[i] = rank;
    });
    return result;
  }

  //: Mirrors spectrum_labels.label_tier -- maps a selectSpectrumLabels rank
  //: to the CSS-visible tier `buildLuckSpectrum` renders.
  function labelTier(rank) {
    if (rank === null || rank === undefined) return "unlabeled";
    return rank === 0 ? "primary" : "secondary";
  }

  function buildLuckSpectrum(markers, domain, sizeClass) {
    var ordered = markers
      .map(function (m) {
        return { m: m, positionPct: spectrumPositionPct(m.value, domain) };
      })
      .sort(function (a, b) {
        return a.positionPct - b.positionPct;
      });
    ordered.forEach(function (entry) {
      entry.m.positionPct = entry.positionPct;
      if (entry.positionPct <= SPECTRUM_EDGE_MARGIN_PCT) entry.m.edge = "start";
      else if (entry.positionPct >= 100 - SPECTRUM_EDGE_MARGIN_PCT) entry.m.edge = "end";
      else entry.m.edge = "mid";
    });

    var zeroPct = spectrumPositionPct(0, domain);

    var wrap = document.createElement("div");
    wrap.className = "luck-spectrum luck-spectrum-" + (sizeClass || "lg");
    wrap.setAttribute("role", "img");
    wrap.setAttribute(
      "aria-label",
      "Luck Spectrum: unfavorable to favorable Contact Luck, with a marker at zero and one marker per plotted play"
    );

    var track = document.createElement("div");
    track.className = "luck-spectrum-track";

    var zero = document.createElement("span");
    zero.className = "luck-spectrum-zero";
    zero.style.left = zeroPct.toFixed(2) + "%";
    zero.setAttribute("aria-hidden", "true");
    track.appendChild(zero);

    markers.forEach(function (m) {
      var favorable = m.value >= 0;
      var el = document.createElement(m.url ? "a" : "div");
      el.className =
        "luck-spectrum-marker luck-spectrum-marker-edge-" +
        m.edge +
        " luck-spectrum-marker-tier-" +
        (m.tier || "secondary") +
        " " +
        (favorable ? "interval-positive" : "interval-negative");
      el.style.left = m.positionPct.toFixed(2) + "%";
      if (m.url) el.href = m.url;
      else el.tabIndex = 0;
      el.title =
        m.label + ": " + formatSigned(m.value) + (m.sublabel ? " (" + m.sublabel + ")" : "");

      var dot = document.createElement("span");
      dot.className = "luck-spectrum-marker-dot";
      dot.setAttribute("aria-hidden", "true");
      el.appendChild(dot);

      var text = document.createElement("span");
      text.className = "luck-spectrum-marker-text";
      var valueEl = document.createElement("span");
      valueEl.className = "luck-spectrum-marker-value";
      valueEl.textContent = formatSigned(m.value);
      text.appendChild(valueEl);
      var labelEl = document.createElement("span");
      labelEl.className = "luck-spectrum-marker-label";
      labelEl.textContent = m.label;
      text.appendChild(labelEl);
      el.appendChild(text);

      track.appendChild(el);
    });
    wrap.appendChild(track);

    var scale = document.createElement("div");
    scale.className = "luck-spectrum-scale";
    scale.setAttribute("aria-hidden", "true");
    var unfav = document.createElement("span");
    unfav.className = "luck-spectrum-scale-label luck-spectrum-scale-unfavorable";
    unfav.textContent = "Unfavorable";
    scale.appendChild(unfav);
    var zeroLabel = document.createElement("span");
    zeroLabel.className = "luck-spectrum-scale-zero-label";
    zeroLabel.style.left = zeroPct.toFixed(2) + "%";
    zeroLabel.textContent = "0";
    scale.appendChild(zeroLabel);
    var fav = document.createElement("span");
    fav.className = "luck-spectrum-scale-label luck-spectrum-scale-favorable";
    fav.textContent = "Favorable";
    scale.appendChild(fav);
    wrap.appendChild(scale);

    return wrap;
  }

  //: How many of the biggest favorable/unfavorable showcase plays each get
  //: PLOTTED on the Luck Spectrum above the two card groups -- kept small
  //: (matches the homepage's own top_n) so there's still a sense of
  //: distribution/clustering, not just the single extreme on each side.
  var SHOWCASE_SPECTRUM_TOP_N = 3;

  //: How many of those plotted plays get a persistent visible LABEL, per
  //: side -- Explore is more aggressive than the homepage (which allows
  //: 2/side) because the cards below already show every play's name/value;
  //: the spectrum here only needs to call out the single most extreme
  //: favorable/unfavorable play. The rest still plot as small unlabeled
  //: dots (hover/focus reveals them) -- see selectSpectrumLabels above.
  var SHOWCASE_SPECTRUM_MAX_LABELS_PER_SIDE = 1;

  function buildSideMarkers(rows, side) {
    var pool = rows
      .slice()
      .sort(function (a, b) {
        return a.rank - b.rank;
      })
      .slice(0, SHOWCASE_SPECTRUM_TOP_N);
    if (!pool.length) return [];
    var values = pool.map(function (r) {
      return r.contact_luck_runs;
    });
    var ranks = selectSpectrumLabels(
      values,
      SHOWCASE_SPECTRUM_MAX_LABELS_PER_SIDE,
      SPECTRUM_MIN_SEPARATION_FRACTION
    );
    return pool.map(function (r, i) {
      return {
        value: r.contact_luck_runs,
        label: playerLabel(r),
        sublabel: "#" + r.rank + " " + side,
        tier: labelTier(ranks[i]),
        url: "/plays/?id=" + encodeURIComponent(r.play_id),
      };
    });
  }

  function renderShowcaseSpectrum(section, favorable, unfavorable) {
    var wrapEl = qs('[data-role="showcase-spectrum-wrap"]', section);
    var mountEl = qs('[data-role="showcase-spectrum"]', section);
    if (!wrapEl || !mountEl) return;
    var markers = buildSideMarkers(favorable, "favorable").concat(
      buildSideMarkers(unfavorable, "unfavorable")
    );
    if (!markers.length) return;
    var values = markers.map(function (m) {
      return m.value;
    });
    var domain = computeSpectrumDomain(values);
    mountEl.innerHTML = "";
    var spectrumEl = buildLuckSpectrum(markers, domain, "lg");
    mountEl.appendChild(spectrumEl);
    // Must come AFTER `hidden = false` -- a `hidden` (display:none)
    // subtree has no real layout, so getBoundingClientRect() inside
    // declutterLuckSpectrum would read zeroed rects and nudge nothing.
    wrapEl.hidden = false;
    declutterLuckSpectrum(spectrumEl.querySelector(".luck-spectrum-track"));
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
        // `bodyEl` (the ancestor `[data-role="showcase-body"]`, wrapping
        // the spectrum mount point) must be unhidden BEFORE
        // renderShowcaseSpectrum runs -- it calls declutterLuckSpectrum,
        // which measures real rendered label positions via
        // getBoundingClientRect(); a `display:none` ancestor makes every
        // element in the subtree report a zeroed rect, silently breaking
        // the whole collision calculation (confirmed via a real-browser
        // check: every "overlap" then looks identical, producing a bogus
        // but plausible-looking incrementing shift instead of an error).
        if (loadingEl) loadingEl.hidden = true;
        if (bodyEl) bodyEl.hidden = false;
        renderShowcaseSpectrum(section, favorable, unfavorable);
        renderShowcaseGroup(section, "favorable", favorable);
        renderShowcaseGroup(section, "unfavorable", unfavorable);
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
    var selectedAvatarEl = qs('[data-role="explore-selected-avatar"]');
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
        var nameCell = document.createElement("div");
        nameCell.className = "leaderboard-name-cell";
        nameCell.appendChild(buildAvatar(row.batter_id, "player-avatar-sm"));
        var link = document.createElement("a");
        link.className = "player-link";
        // Query-param route (Phase 4.1: a single static /plays/ shell,
        // never one directory per play_id). Absolute path, matching this
        // site's existing routing convention (root_prefix is always "/").
        link.href = "/plays/?id=" + encodeURIComponent(row.play_id);
        link.textContent = row.batter_name || "Player " + row.batter_id;
        nameCell.appendChild(link);
        batterTd.appendChild(nameCell);
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
      if (selectedAvatarEl) {
        selectedAvatarEl.innerHTML = "";
        selectedAvatarEl.appendChild(buildAvatar(entry.batter_id, "player-avatar-lg"));
      }
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
        searchInput.disabled = false;

        // Optional `?batter=<batter_id>` deep link (e.g. from a player
        // page's "Explore this player's plays" link) -- auto-selects that
        // hitter instead of showing the empty-state prompt, IF that
        // batter_id is actually present in this catalog. Never a hard
        // requirement: a missing/unknown/malformed id just falls back to
        // the ordinary empty-state prompt, exactly as a bare /explore/
        // visit already does.
        var requestedBatterId = new URLSearchParams(window.location.search).get("batter");
        var preselected = null;
        if (requestedBatterId !== null) {
          var requestedId = parseInt(requestedBatterId, 10);
          preselected = playersCatalog.find(function (entry) {
            return entry.batter_id === requestedId;
          });
        }
        if (preselected) {
          selectPlayer(preselected);
        } else {
          promptEl.hidden = false;
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
