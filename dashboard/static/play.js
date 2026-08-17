// Contact Luck v1.4.0 -- individual play detail page (Phase 4.1: ONE static
// shell for every play, never one HTML directory per play_id -- see
// dashboard/build.py's own comment on why). This page's static HTML embeds
// NOTHING play-specific; on load, play.js reads `?id=<play_id>` from the
// URL itself, validates its shape, derives `game_pk` from that SAME stable
// identity (a play_id is always `game_pk-at_bat_number-pitch_number`, so
// game_pk never needs a separate lookup or the full search index), and
// fetches ONLY that one game's ALREADY-COMPUTED, committed per-game detail
// JSON (see dashboard/explore_content.py / demo/build_play_explorer_
// fixture.py) -- never search-index.json on a direct play-page visit. No
// scoring formula, probability, or Contact Luck value is recomputed here.
// A malformed id, a fetch failure, or a play_id absent from its game file
// all render the SAME controlled "Play not found" state, never a broken or
// blank page.
(function () {
  "use strict";

  var CLASS_ORDER = ["out", "single", "double", "triple", "home_run"];
  var CLASS_LABELS = { out: "Out", single: "1B", double: "2B", triple: "3B", home_run: "HR" };
  var OUTCOME_LABELS = {
    out: "Out",
    single: "Single",
    double: "Double",
    triple: "Triple",
    home_run: "Home run",
  };

  // A play_id is ALWAYS exactly `<game_pk>-<at_bat_number>-<pitch_number>`
  // (mlb_luck_score.data.clean_batted_balls.build_event_id) -- three
  // non-negative integers separated by hyphens, nothing else. Validated
  // BEFORE any fetch is attempted, so a malformed id never reaches the
  // network layer at all.
  var PLAY_ID_PATTERN = /^(\d+)-\d+-\d+$/;

  // Illustrative-only field geometry -- same visual language (fence arc,
  // foul lines, accent-field color) as dashboard/visuals.py's field SVGs,
  // but computed here in plain JS from the play's own recorded
  // spray_angle_approx/hit_distance_sc (a fixed distance-to-radius mapping,
  // NOT exit-velocity/launch-angle modulated like the "Try It Yourself"
  // simulator's depth curve -- this page shows one fixed real play, not an
  // interactive slider). Never a physics simulation; never implies a
  // reconstruction of the actual play.
  var FIELD_HOME_X = 150;
  var FIELD_HOME_Y = 260;
  var FIELD_FENCE_RADIUS = 215;
  var FIELD_MAX_SPRAY_DEG = 45;
  var FIELD_MAX_DISTANCE_FT = 450;
  var FIELD_MIN_DEPTH_FRACTION = 0.08;

  function qs(selector, root) {
    return (root || document).querySelector(selector);
  }

  function formatSigned(value) {
    if (value === null || value === undefined) return "—";
    var sign = value >= 0 ? "+" : "";
    return sign + value.toFixed(2);
  }

  function outcomeLabel(cls) {
    return OUTCOME_LABELS[cls] || cls;
  }

  function renderField(page, play) {
    var svg = qs('[data-role="play-field-svg"]', page);
    var ball = qs('[data-role="play-field-ball"]', page);
    if (!svg || !ball) return;
    if (play.spray_angle_approx === null || play.hit_distance_sc === null) {
      ball.setAttribute("cx", String(FIELD_HOME_X));
      ball.setAttribute("cy", String(FIELD_HOME_Y));
      return;
    }
    var depthFraction = Math.max(
      FIELD_MIN_DEPTH_FRACTION,
      Math.min(1, play.hit_distance_sc / FIELD_MAX_DISTANCE_FT)
    );
    var radius = depthFraction * FIELD_FENCE_RADIUS;
    var clampedSpray = Math.max(
      -FIELD_MAX_SPRAY_DEG,
      Math.min(FIELD_MAX_SPRAY_DEG, play.spray_angle_approx)
    );
    var angleRad = (clampedSpray * Math.PI) / 180;
    var endX = FIELD_HOME_X + radius * Math.sin(angleRad);
    var endY = FIELD_HOME_Y - radius * Math.cos(angleRad);
    ball.setAttribute("cx", endX.toFixed(1));
    ball.setAttribute("cy", endY.toFixed(1));
    ball.classList.add(play.contact_luck_runs >= 0 ? "demo-ball-favorable" : "demo-ball-unfavorable");
  }

  function renderProbabilities(page, play) {
    var container = qs('[data-role="play-prob-bars"]', page);
    if (!container) return;
    container.innerHTML = "";
    CLASS_ORDER.forEach(function (cls) {
      var p = play["p_" + cls];
      var pct = p === null || p === undefined ? 0 : p * 100;
      var isObserved = cls === play.outcome_class;

      var row = document.createElement("div");
      row.className = "demo-prob-row" + (isObserved ? " demo-prob-row-observed" : "");

      var label = document.createElement("span");
      label.className = "demo-prob-label";
      label.textContent = CLASS_LABELS[cls] || cls;
      row.appendChild(label);

      var track = document.createElement("span");
      track.className = "demo-prob-bar-track";
      var fill = document.createElement("span");
      fill.className = "demo-prob-bar-fill";
      fill.style.width = pct.toFixed(2) + "%";
      track.appendChild(fill);
      row.appendChild(track);

      var pctEl = document.createElement("span");
      pctEl.className = "demo-prob-pct";
      pctEl.textContent = pct.toFixed(1) + "%";
      row.appendChild(pctEl);

      container.appendChild(row);
    });
  }

  function renderPlay(page, play) {
    var favorable = play.contact_luck_runs >= 0;

    document.title = (play.batter_name || "Player " + play.batter_id) + " — Play — Contact Luck";

    var nameEl = qs('[data-role="play-batter-name"]', page);
    if (nameEl) nameEl.textContent = play.batter_name || "Player " + play.batter_id;

    var metaEl = qs('[data-role="play-meta"]', page);
    if (metaEl) {
      metaEl.textContent =
        "MLBAM ID " + play.batter_id + " · " + play.game_date + " · Game " + play.game_pk;
    }

    var badgeEl = qs('[data-role="play-luck-badge"]', page);
    if (badgeEl) {
      badgeEl.textContent = favorable ? "Favorable" : "Unfavorable";
      badgeEl.className =
        "demo-luck-badge " + (favorable ? "demo-luck-badge-favorable" : "demo-luck-badge-unfavorable");
    }

    var evEl = qs('[data-role="play-ev"]', page);
    if (evEl) {
      evEl.textContent =
        play.launch_speed === null || play.launch_speed === undefined
          ? "—"
          : play.launch_speed.toFixed(1) + " mph";
    }

    var laEl = qs('[data-role="play-la"]', page);
    if (laEl) {
      laEl.textContent =
        play.launch_angle === null || play.launch_angle === undefined
          ? "—"
          : Math.round(play.launch_angle) + "°";
    }

    var bbTypeEl = qs('[data-role="play-bb-type"]', page);
    if (bbTypeEl) {
      bbTypeEl.textContent = play.bb_type ? String(play.bb_type).replace(/_/g, " ") : "—";
    }

    var outcomeEl = qs('[data-role="play-outcome"]', page);
    if (outcomeEl) outcomeEl.textContent = outcomeLabel(play.outcome_class);

    renderProbabilities(page, play);

    var expectedEl = qs('[data-role="play-expected-rv"]', page);
    if (expectedEl) expectedEl.textContent = formatSigned(play.expected_run_value);

    var observedEl = qs('[data-role="play-observed-rv"]', page);
    if (observedEl) observedEl.textContent = formatSigned(play.observed_run_value);

    var luckFigureEl = qs('[data-role="play-luck-figure"]', page);
    if (luckFigureEl) {
      luckFigureEl.textContent = formatSigned(play.contact_luck_runs) + " runs";
      luckFigureEl.className =
        "demo-luck-figure " + (favorable ? "interval-positive" : "interval-negative");
    }

    renderField(page, play);
  }

  function showNotFound(page) {
    var loadingEl = qs('[data-role="play-loading"]', page);
    var notFoundEl = qs('[data-role="play-not-found"]', page);
    if (loadingEl) loadingEl.hidden = true;
    if (notFoundEl) notFoundEl.hidden = false;
  }

  function initPlayPage() {
    var page = qs('[data-role="play-page"]');
    if (!page) return;

    var params = new URLSearchParams(window.location.search);
    var playId = params.get("id");

    if (!playId) {
      showNotFound(page);
      return;
    }
    var match = PLAY_ID_PATTERN.exec(playId);
    if (!match) {
      // Malformed play_id -- never even attempt a fetch.
      showNotFound(page);
      return;
    }
    var gamePk = match[1];

    fetch("/explore/games/" + encodeURIComponent(gamePk) + ".json")
      .then(function (response) {
        if (!response.ok) throw new Error("failed to load game detail: " + response.status);
        return response.json();
      })
      .then(function (plays) {
        var play = null;
        for (var i = 0; i < plays.length; i++) {
          if (plays[i].play_id === playId) {
            play = plays[i];
            break;
          }
        }
        if (!play) {
          showNotFound(page);
          return;
        }
        renderPlay(page, play);
        var loadingEl = qs('[data-role="play-loading"]', page);
        var bodyEl = qs('[data-role="play-body"]', page);
        if (loadingEl) loadingEl.hidden = true;
        if (bodyEl) bodyEl.hidden = false;
      })
      .catch(function (err) {
        showNotFound(page);
        if (window.console && window.console.error) window.console.error(err);
      });
  }

  document.addEventListener("DOMContentLoaded", initPlayPage);
})();
