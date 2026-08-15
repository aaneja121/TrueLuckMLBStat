// Contact Luck v1.3.1 -- "Try It Yourself" counterfactual EV/launch-angle
// simulator. Purely presentational: fetches the ALREADY-COMPUTED,
// committed counterfactual grid ONCE on page load (a plain static JSON
// file -- see dashboard/demo_counterfactual_content.py), then does direct
// array-index lookups into it. No interpolation, no inference, no network
// request on slider movement. Entirely independent of static/demo.js (the
// v1.3.0 walkthrough) -- no shared state, no shared DOM, loaded alongside
// it via its own <script> tag.
(function () {
  "use strict";

  var CLASS_ORDER = ["out", "single", "double", "triple", "home_run"];
  var GRID_URL = "counterfactual-grid.json";

  function qs(selector, root) {
    return (root || document).querySelector(selector);
  }
  function qsa(selector, root) {
    return Array.prototype.slice.call((root || document).querySelectorAll(selector));
  }

  function formatSigned(value) {
    var sign = value >= 0 ? "+" : "";
    return sign + value.toFixed(2);
  }

  function expectedRunValue(cell, runValueTable) {
    var total = 0;
    for (var i = 0; i < CLASS_ORDER.length; i++) {
      total += cell[i] * runValueTable[CLASS_ORDER[i]];
    }
    return total;
  }

  function humanizeBbType(bbType) {
    return String(bbType).replace(/_/g, " ");
  }

  // Illustrative field-visualization heuristic (Version 1.3.1 field-view
  // addition). This is NOT a physics simulation and must never be treated
  // as one -- there is no ball-flight model anywhere in this codebase (see
  // CLAUDE.md's rejection of a calibrated vacuum-projectile hit_distance_sc
  // model for this exact simulator), and this function does not read or
  // touch `hit_distance_sc`/`spray_angle_approx` as MODEL features -- it
  // only borrows the fixed spray angle already displayed in the "Fixed
  // from the real play" line to decide which direction the illustrative
  // ball travels. Exit velocity and launch angle are normalized against
  // fixed, hand-picked illustrative bounds (not each play's own slider
  // range) specifically so "low"/"high" reads consistently regardless of
  // which reference play is selected.
  var VIS_EV_MIN_MPH = 40;
  var VIS_EV_MAX_MPH = 115;
  var VIS_MIN_DEPTH_FRACTION = 0.08;
  var VIS_MAX_DEPTH_FRACTION = 0.97;
  var VIS_MIN_ARC_LIFT_PX = 20;
  var VIS_MAX_ARC_LIFT_PX = 205;

  function evDepthFactor(evMph) {
    var t = (evMph - VIS_EV_MIN_MPH) / (VIS_EV_MAX_MPH - VIS_EV_MIN_MPH);
    return Math.max(0, Math.min(1, t));
  }

  // A smooth, symmetric bell-shaped curve peaking at 45 degrees and zero at
  // 0/90 -- used ONLY to mildly modulate DEPTH below, never arc height (see
  // computeFieldVisualState). An illustrative shaping curve only, loosely
  // evoking the ordinary intuition that a very flat or very steep launch
  // angle carries less distance than a medium one; not derived from, and
  // not claiming to reproduce, actual projectile physics or this model's
  // own predictions.
  function laDepthShapeFactor(laDeg) {
    var clamped = Math.max(0, Math.min(90, laDeg));
    var rad = (clamped * Math.PI) / 180;
    return Math.sin(2 * rad);
  }

  // Monotonically increasing in launch angle across the FULL supported
  // range (0-90 degrees), deliberately NOT the bell-shaped curve above --
  // arc height and depth read as different physical quantities (how high
  // vs. how far), so they must never share one scalar. A steep 80-87
  // degree popup must always render as the TALLEST illustrative arc, never
  // a short one, which a "carry distance" bell curve would get backwards.
  function arcHeightFactor(laDeg) {
    var clamped = Math.max(0, Math.min(90, laDeg));
    return clamped / 90;
  }

  function computeFieldVisualState(evMph, laDeg, sprayDeg, field) {
    // Exit velocity is the PRIMARY driver of depth; launch angle only
    // mildly modulates it (0.7x-1.0x, bell-shaped) so a flat, hard-hit
    // line drive still reads as reasonably deep, not short, while a very
    // high launch angle reads as shallower than a productive mid-angle
    // hit at the same exit velocity.
    var depthFactor = evDepthFactor(evMph) * (0.7 + 0.3 * laDepthShapeFactor(laDeg));
    var depthFraction =
      VIS_MIN_DEPTH_FRACTION + (VIS_MAX_DEPTH_FRACTION - VIS_MIN_DEPTH_FRACTION) * depthFactor;
    var radius = depthFraction * field.fenceRadius;

    var clampedSpray = Math.max(-field.maxSprayDeg, Math.min(field.maxSprayDeg, sprayDeg));
    var angleRad = (clampedSpray * Math.PI) / 180;
    var endX = field.homeX + radius * Math.sin(angleRad);
    var endY = field.homeY - radius * Math.cos(angleRad);

    // Arc height is a SEPARATE scalar, driven only by launch angle and
    // anchored PURELY from home plate's own height -- never from `endY`.
    // An earlier version anchored the lift against `Math.min(homeY, endY)`
    // (or clamped against it as a floor), which reintroduces exactly the
    // coupling this rewrite exists to remove: because the depth bell curve
    // above pulls the endpoint back toward home at high launch angles, any
    // dependency on `endY` -- even just as a one-sided clamp -- can make an
    // 80-87 degree popup render with a LOWER arc than a 45-60 degree hit,
    // since the clamp binds across a range that straddles the depth curve's
    // own peak. Anchoring purely from home plate removes that coupling
    // entirely: `controlY` here is a pure, strictly monotonic function of
    // launch angle alone, guaranteed never to depend on exit velocity or
    // depth. The tradeoff is a purely cosmetic one -- a very deep, very
    // flat (low-launch-angle, high-exit-velocity) shot's control point can
    // sit closer to home than its own endpoint, which still renders as a
    // sensible shallow-then-flattening curve (verified visually), not a
    // broken shape.
    var liftPx =
      VIS_MIN_ARC_LIFT_PX + (VIS_MAX_ARC_LIFT_PX - VIS_MIN_ARC_LIFT_PX) * arcHeightFactor(laDeg);
    var controlX = (field.homeX + endX) / 2;
    var controlY = field.homeY - liftPx;

    return { endX: endX, endY: endY, controlX: controlX, controlY: controlY };
  }

  function initSimulator(section, gridData) {
    var runValueTable = gridData.model_configuration.run_value_table;
    var referencePlays = gridData.reference_plays;

    var loadingEl = qs('[data-role="simulator-loading"]', section);
    var bodyEl = qs('[data-role="simulator-body"]', section);
    var evSlider = qs('[data-role="simulator-ev-slider"]', section);
    var laSlider = qs('[data-role="simulator-la-slider"]', section);
    var evValueEl = qs('[data-role="simulator-ev-value"]', section);
    var laValueEl = qs('[data-role="simulator-la-value"]', section);
    var fixedContextEl = qs('[data-role="simulator-fixed-context"]', section);
    var probRows = qsa('[data-role="simulator-prob-row"]', section);
    var expectedRvEls = qsa('[data-role="simulator-expected-rv"]', section);
    var observedRvEl = qs('[data-role="simulator-observed-rv"]', section);
    var luckBadgeEl = qs('[data-role="simulator-luck-badge"]', section);
    var luckFigureEl = qs('[data-role="simulator-luck-figure"]', section);
    var luckExplanationEl = qs('[data-role="simulator-luck-explanation"]', section);
    var outcomeButtons = qsa('[data-role="simulator-outcome-button"]', section);
    var playButtons = qsa('[data-role="simulator-select-play"]', section);
    var resetButton = qs('[data-role="simulator-reset"]', section);

    var fieldSvg = qs('[data-role="simulator-field-svg"]', section);
    var fieldPath = qs('[data-role="simulator-field-path"]', section);
    var fieldBall = qs('[data-role="simulator-field-ball"]', section);
    var field = fieldSvg
      ? {
          homeX: parseFloat(fieldSvg.getAttribute("data-home-x")),
          homeY: parseFloat(fieldSvg.getAttribute("data-home-y")),
          fenceRadius: parseFloat(fieldSvg.getAttribute("data-fence-radius")),
          maxSprayDeg: parseFloat(fieldSvg.getAttribute("data-max-spray-deg")),
        }
      : null;

    var state = { exampleId: null, evIndex: 0, laIndex: 0, selectedOutcome: null };

    function currentPlay() {
      return referencePlays[state.exampleId];
    }

    function currentCell() {
      var play = currentPlay();
      var rowIndex = state.evIndex * play.grid_shape.n_la + state.laIndex;
      return play.grid[rowIndex];
    }

    function render() {
      var play = currentPlay();
      var evValue = play.exit_velocity_values[state.evIndex];
      var laValue = play.launch_angle_values[state.laIndex];

      evSlider.max = String(play.exit_velocity_values.length - 1);
      evSlider.value = String(state.evIndex);
      evSlider.setAttribute("aria-valuetext", evValue + " mph");
      evValueEl.textContent = evValue + " mph";

      laSlider.max = String(play.launch_angle_values.length - 1);
      laSlider.value = String(state.laIndex);
      laSlider.setAttribute("aria-valuetext", laValue + "°");
      laValueEl.textContent = laValue + "°";

      var cell = currentCell();
      probRows.forEach(function (row) {
        var cls = row.getAttribute("data-outcome-class");
        var idx = CLASS_ORDER.indexOf(cls);
        var pct = cell[idx] * 100;
        var fill = qs('[data-role="simulator-prob-fill"]', row);
        var pctEl = qs('[data-role="simulator-prob-pct"]', row);
        fill.style.setProperty("--demo-target-width", pct.toFixed(2) + "%");
        pctEl.textContent = pct.toFixed(1) + "%";
      });

      var expectedRv = expectedRunValue(cell, runValueTable);
      expectedRvEls.forEach(function (el) {
        el.textContent = formatSigned(expectedRv);
      });

      var fixed = play.fixed_context;
      fixedContextEl.textContent =
        "Fixed from the real play: " +
        humanizeBbType(fixed.bb_type) +
        " contact, spray angle " +
        fixed.spray_angle_approx.toFixed(1) +
        "°, hit distance " +
        Math.round(fixed.hit_distance_sc) +
        " ft, batter stance " +
        fixed.stand +
        ".";

      if (field && fieldPath && fieldBall) {
        var visual = computeFieldVisualState(evValue, laValue, fixed.spray_angle_approx, field);
        fieldPath.setAttribute(
          "d",
          "M " +
            field.homeX.toFixed(1) +
            "," +
            field.homeY.toFixed(1) +
            " Q " +
            visual.controlX.toFixed(1) +
            "," +
            visual.controlY.toFixed(1) +
            " " +
            visual.endX.toFixed(1) +
            "," +
            visual.endY.toFixed(1)
        );
        fieldBall.setAttribute("cx", visual.endX.toFixed(1));
        fieldBall.setAttribute("cy", visual.endY.toFixed(1));
      }

      outcomeButtons.forEach(function (btn) {
        var selected = btn.getAttribute("data-outcome-class") === state.selectedOutcome;
        btn.setAttribute("aria-pressed", selected ? "true" : "false");
        btn.classList.toggle("simulator-outcome-button-selected", selected);
      });

      var observedRv = runValueTable[state.selectedOutcome];
      observedRvEl.textContent = formatSigned(observedRv);

      var contactLuck = observedRv - expectedRv;
      var favorable = contactLuck >= 0;
      luckBadgeEl.textContent = favorable ? "Favorable" : "Unfavorable";
      luckBadgeEl.className =
        "demo-luck-badge " + (favorable ? "demo-luck-badge-favorable" : "demo-luck-badge-unfavorable");
      luckFigureEl.textContent = formatSigned(contactLuck) + " runs";
      luckFigureEl.className = "demo-luck-figure " + (favorable ? "interval-positive" : "interval-negative");
      if (luckExplanationEl) {
        luckExplanationEl.textContent = favorable
          ? "The recorded outcome was worth more than the contact was expected to produce."
          : "The contact was worth more than the recorded outcome.";
      }

      playButtons.forEach(function (btn) {
        var selected = btn.getAttribute("data-example-id") === state.exampleId;
        btn.setAttribute("aria-selected", selected ? "true" : "false");
        btn.classList.toggle("simulator-play-button-selected", selected);
      });
    }

    function selectPlay(exampleId) {
      var play = referencePlays[exampleId];
      if (!play) return;
      state.exampleId = exampleId;
      state.evIndex = play.original_grid_index.ev_index;
      state.laIndex = play.original_grid_index.la_index;
      state.selectedOutcome = play.original_outcome_class;
      render();
    }

    evSlider.addEventListener("input", function () {
      state.evIndex = parseInt(evSlider.value, 10);
      render();
    });
    laSlider.addEventListener("input", function () {
      state.laIndex = parseInt(laSlider.value, 10);
      render();
    });
    outcomeButtons.forEach(function (btn) {
      btn.addEventListener("click", function () {
        state.selectedOutcome = btn.getAttribute("data-outcome-class");
        render();
      });
    });
    playButtons.forEach(function (btn) {
      btn.addEventListener("click", function () {
        selectPlay(btn.getAttribute("data-example-id"));
      });
    });
    resetButton.addEventListener("click", function () {
      selectPlay(state.exampleId);
    });

    var initialId = playButtons.length
      ? playButtons[0].getAttribute("data-example-id")
      : Object.keys(referencePlays)[0];
    selectPlay(initialId);

    if (loadingEl) loadingEl.hidden = true;
    bodyEl.hidden = false;
  }

  function initSimulatorSection() {
    var section = document.querySelector('[data-role="counterfactual-simulator"]');
    if (!section) return;

    fetch(GRID_URL)
      .then(function (response) {
        if (!response.ok) {
          throw new Error("failed to load counterfactual grid: " + response.status);
        }
        return response.json();
      })
      .then(function (gridData) {
        initSimulator(section, gridData);
      })
      .catch(function (err) {
        var loadingEl = qs('[data-role="simulator-loading"]', section);
        if (loadingEl) {
          loadingEl.textContent = "This interactive section could not load. The rest of the page is unaffected.";
        }
        if (window.console && window.console.error) {
          window.console.error(err);
        }
      });
  }

  document.addEventListener("DOMContentLoaded", initSimulatorSection);
})();
