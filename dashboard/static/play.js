// Contact Luck -- individual play detail page (redesign Phase 6).
//
// Phase 4.1 routing, unchanged: ONE static shell for every play, never one
// HTML directory per play_id (see dashboard/build.py's own comment on why).
// The shell embeds NOTHING play-specific. On load this file reads
// `?id=<play_id>` from the URL, validates its shape, derives `game_pk` from
// that SAME stable identity (a play_id is always
// `game_pk-at_bat_number-pitch_number`), and fetches ONLY that one game's
// already-computed per-game JSON. No scoring formula, probability or
// Contact Luck value is recomputed here. A malformed id, a fetch failure,
// or a play_id absent from its game file all render the SAME controlled
// "Play not found" state, never a broken or blank page.
//
// Phase 6 adds one thing: the expected-versus-observed figure, drawn on the
// build-supplied `run_value` domain that /explore/ is drawn on. The domain
// arrives as data; this file interpolates a layout percentage against it
// and derives no scale of its own.
(function () {
  "use strict";

  var CLASS_ORDER = ["out", "single", "double", "triple", "home_run"];
  var OUTCOME_LABELS = {
    out: "Out",
    single: "Single",
    double: "Double",
    triple: "Triple",
    home_run: "Home run",
  };

  // Illustrative-only field geometry: a fixed distance-to-radius mapping
  // from the play's own recorded spray_angle_approx/hit_distance_sc. Never
  // a physics simulation, never a reconstruction of the actual ball flight.
  var FIELD_HOME_X = 150;
  var FIELD_HOME_Y = 230;
  var FIELD_FENCE_RADIUS = 215;
  var FIELD_MAX_SPRAY_DEG = 45;
  var FIELD_MAX_DISTANCE_FT = 450;
  var FIELD_MIN_DEPTH_FRACTION = 0.08;

  var EM_DASH = "—";

  var shared = window.ContactLuck || {};
  var formatSigned = shared.formatSigned;
  var runValuePct = shared.runValuePct;

  function qs(selector, root) {
    return (root || document).querySelector(selector);
  }

  function setText(page, role, text) {
    var el = qs('[data-role="' + role + '"]', page);
    if (el) el.textContent = text;
  }

  function outcomeLabel(cls) {
    return OUTCOME_LABELS[cls] || cls;
  }

  function playerLabel(play) {
    return play.batter_name || "Player " + play.batter_id;
  }

  function bbTypeLabel(bbType) {
    return bbType ? String(bbType).replace(/_/g, " ") : EM_DASH;
  }

  function evText(play) {
    return play.launch_speed === null || play.launch_speed === undefined
      ? EM_DASH
      : play.launch_speed.toFixed(1) + " mph";
  }

  function laText(play) {
    return play.launch_angle === null || play.launch_angle === undefined
      ? EM_DASH
      : Math.round(play.launch_angle) + "°";
  }

  // ── The run-value figure ────────────────────────────────────────────────
  // One axis, two marks and the distance between them. Expected is a hollow
  // ring above the field's centre line, observed is a solid dot; each label
  // sits on its own side of the field with its own value, so the pair stays
  // readable when the two marks nearly touch and neither position is ever
  // nudged to make room. Direction is carried by the labels and the mark
  // shapes as well as by colour.
  function renderGap(page, play, scale) {
    var field = qs('[data-role="play-gap-field"]', page);
    if (!field || !scale) return;

    var expected = play.expected_run_value;
    var observed = play.observed_run_value;
    var favorable = play.contact_luck_runs >= 0;

    var expectedPct = runValuePct(scale, expected);
    var observedPct = runValuePct(scale, observed);

    field.style.setProperty("--play-expected", expectedPct);
    field.style.setProperty("--play-observed", observedPct);
    field.className =
      "cl-scale-field play-gap-field " +
      (favorable ? "cl-scale-favorable" : "cl-scale-unfavorable");
    field.setAttribute(
      "aria-label",
      "Run value on this play. The model expected " +
        formatSigned(expected) +
        " runs; the play actually produced " +
        formatSigned(observed) +
        " runs. The gap between them is " +
        formatSigned(play.contact_luck_runs) +
        " runs of Contact Luck, " +
        (favorable ? "in the hitter's favor." : "against the hitter.")
    );

    // Each label is anchored at its own mark. Near an edge the anchor
    // changes, never the position.
    anchorLabel(qs('[data-role="play-gap-expected-label"]', page), expectedPct);
    anchorLabel(qs('[data-role="play-gap-observed-label"]', page), observedPct);
  }

  function anchorLabel(el, pct) {
    if (!el) return;
    var fraction = parseFloat(pct) / 100;
    el.style.setProperty("--play-mark", pct);
    el.dataset.anchor = fraction < 0.14 ? "start" : fraction > 0.86 ? "end" : "center";
  }

  // ── The field diagram ───────────────────────────────────────────────────
  // Shown only when the play carries both a spray angle and a hit distance.
  // Where either is missing the figure stays hidden rather than drawing a
  // ball at home plate, which would read as a measurement.
  function renderField(page, play) {
    var wrap = qs('[data-role="play-field-wrap"]', page);
    var ball = qs('[data-role="play-field-ball"]', page);
    if (!wrap || !ball) return;

    var hasLocation =
      play.spray_angle_approx !== null &&
      play.spray_angle_approx !== undefined &&
      play.hit_distance_sc !== null &&
      play.hit_distance_sc !== undefined;
    if (!hasLocation) {
      wrap.hidden = true;
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
    ball.setAttribute("cx", (FIELD_HOME_X + radius * Math.sin(angleRad)).toFixed(1));
    ball.setAttribute("cy", (FIELD_HOME_Y - radius * Math.cos(angleRad)).toFixed(1));
    ball.classList.add(
      play.contact_luck_runs >= 0 ? "play-field-ball-favorable" : "play-field-ball-unfavorable"
    );

    setText(
      page,
      "play-field-caption",
      "Roughly where it landed: " +
        Math.round(play.hit_distance_sc) +
        " feet, from the recorded spray angle. Illustrative, not a reconstruction of the ball flight."
    );
    wrap.hidden = false;
  }

  // ── The model's expectation ─────────────────────────────────────────────
  // Probabilities are bounded 0-100%, so they get a plain horizontal bar
  // against a full-width track and no zero spine. Every class keeps its
  // exact value and its position in the frozen order.
  function renderProbabilities(page, play) {
    var tbody = qs('[data-role="play-prob-rows"]', page);
    if (!tbody) return;
    tbody.innerHTML = "";

    var best = null;
    CLASS_ORDER.forEach(function (cls) {
      var p = play["p_" + cls];
      if (p === null || p === undefined) return;
      if (!best || p > best.p) best = { cls: cls, p: p };
    });

    CLASS_ORDER.forEach(function (cls) {
      var p = play["p_" + cls];
      var pct = p === null || p === undefined ? 0 : p * 100;
      var isObserved = cls === play.outcome_class;

      var tr = document.createElement("tr");
      if (isObserved) tr.className = "is-observed";

      var th = document.createElement("th");
      th.setAttribute("scope", "row");
      th.className = "col-prob-outcome";
      th.textContent = outcomeLabel(cls);
      if (isObserved) {
        var tag = document.createElement("span");
        tag.className = "play-prob-tag";
        tag.textContent = "what happened";
        th.appendChild(tag);
      }
      tr.appendChild(th);

      var barTd = document.createElement("td");
      barTd.className = "col-prob-bar";
      var track = document.createElement("span");
      track.className = "play-prob-track";
      track.setAttribute("aria-hidden", "true");
      var fill = document.createElement("span");
      fill.className = "play-prob-fill";
      fill.style.width = pct.toFixed(2) + "%";
      track.appendChild(fill);
      barTd.appendChild(track);
      tr.appendChild(barTd);

      var pctTd = document.createElement("td");
      pctTd.className = "col-prob-pct num";
      pctTd.textContent = pct < 0.05 && pct > 0 ? "<0.1%" : pct.toFixed(1) + "%";
      tr.appendChild(pctTd);

      tbody.appendChild(tr);
    });

    setText(
      page,
      "play-probabilities-caption",
      "The model's chance of each outcome for this contact, with the recorded result marked."
    );

    if (best) {
      var observedLabel = outcomeLabel(play.outcome_class).toLowerCase();
      var bestLabel = outcomeLabel(best.cls).toLowerCase();
      var pct = (best.p * 100).toFixed(0);
      setText(
        page,
        "play-expectation-lede",
        best.cls === play.outcome_class
          ? "The model gave " +
              bestLabel +
              " the best chance at " +
              pct +
              "%, and that is what happened."
          : "The model gave " +
              bestLabel +
              " the best chance at " +
              pct +
              "%. The play went for a " +
              observedLabel +
              "."
      );
    }
  }

  function renderPlay(page, play, scale) {
    var favorable = play.contact_luck_runs >= 0;
    var name = playerLabel(play);

    document.title = name + " · Play · Contact Luck";

    setText(page, "play-batter-name", name);
    setText(
      page,
      "play-headline",
      outcomeLabel(play.outcome_class) + " · " + play.game_date
    );

    var back = qs('[data-role="play-back-link"]', page);
    if (back) {
      // Return to THIS hitter's plays, not a cold Explore.
      back.setAttribute("href", "/explore/?batter=" + encodeURIComponent(String(play.batter_id)));
      back.textContent = "← " + name + " in Play Explorer";
    }
    var playerLink = qs('[data-role="play-player-link"]', page);
    if (playerLink) {
      playerLink.setAttribute(
        "href",
        "/players/" + encodeURIComponent(String(play.batter_id)) + "/"
      );
    }

    setText(page, "play-luck", formatSigned(play.contact_luck_runs) + " runs");
    var luckEl = qs('[data-role="play-luck"]', page);
    if (luckEl) {
      luckEl.className =
        "play-luck num " + (favorable ? "cl-scale-favorable" : "cl-scale-unfavorable");
    }
    setText(
      page,
      "play-luck-label",
      favorable ? "Favorable Contact Luck" : "Unfavorable Contact Luck"
    );

    setText(page, "play-expected-rv", formatSigned(play.expected_run_value));
    setText(page, "play-observed-rv", formatSigned(play.observed_run_value));

    setText(
      page,
      "play-equation",
      "The model expected " +
        formatSigned(play.expected_run_value) +
        " runs from this contact. The play produced " +
        formatSigned(play.observed_run_value) +
        ". The " +
        formatSigned(play.contact_luck_runs) +
        " gap is the Contact Luck on the play."
    );

    setText(page, "play-ev", evText(play));
    setText(page, "play-la", laText(play));
    setText(page, "play-bb-type", bbTypeLabel(play.bb_type));
    setText(page, "play-outcome", outcomeLabel(play.outcome_class));

    renderGap(page, play, scale);
    renderProbabilities(page, play);
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
    // The play-id contract lives once, in app.js (`gamePkFromPlayId`); the
    // local fallback keeps this page working if app.js failed to load.
    var gamePk = shared.gamePkFromPlayId
      ? shared.gamePkFromPlayId(playId)
      : (/^(\d+)-\d+-\d+$/.exec(playId) || [])[1] || null;
    if (!gamePk) {
      // Malformed play_id -- never even attempt a fetch.
      showNotFound(page);
      return;
    }
    var scale = shared.runValueScale ? shared.runValueScale() : null;

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
        renderPlay(page, play, scale);
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
