// Contact Luck v1.4.1 -- Showcase Play "What if?" sensitivity module for
// the individual play page. Purely presentational: fetches the ALREADY
// -COMPUTED, static showcase-sensitivity/<play_id>.json grid (produced
// offline by demo/build_showcase_sensitivity.py, which owns the mandatory
// canonical-reconciliation gate -- this file never imports or runs any
// model code), then does direct array-index lookups into it on slider
// movement. No interpolation, no inference, no network request after the
// one initial fetch. Entirely independent of dashboard/static/play.js (own
// play_id parsing, own DOM, own state) -- the same "no shared state"
// convention dashboard/static/demo_simulator.js already established
// relative to dashboard/static/demo.js.
//
// Sliders change ONLY the frozen model's outcome-probability distribution
// and its resulting Expected RV for a HYPOTHETICAL exit velocity/launch
// angle -- see "Actual Expected RV" / "Modified Expected RV" / "Change in
// Expected RV" below. They never touch, recompute, or relabel the real
// play's own Recorded result / Final observed RV / Contact Luck (rendered
// separately, above, by play.js): this module never invents a hypothetical
// observed outcome or a hypothetical Contact Luck headline.
//
// A play with no showcase-sensitivity/<play_id>.json (not interactive
// -eligible, or the file simply isn't there) leaves this section hidden --
// never a disabled/broken simulator.
(function () {
  "use strict";

  var CLASS_ORDER = ["out", "single", "double", "triple", "home_run"];
  // The same outcome names the play page's own probability table uses --
  // one quantity, one vocabulary, on one page.
  var CLASS_LABELS = {
    out: "Out",
    single: "Single",
    double: "Double",
    triple: "Triple",
    home_run: "Home run",
  };

  function qs(selector, root) {
    return (root || document).querySelector(selector);
  }

  // Phase 6 consolidation: the signed-number primitive lives once, in
  // app.js, on `window.ContactLuck`.
  var formatSigned = (window.ContactLuck || {}).formatSigned;

  function humanizeBbType(bbType) {
    return String(bbType).replace(/_/g, " ");
  }

  function expectedRunValue(cell, runValueTable) {
    var total = 0;
    for (var i = 0; i < CLASS_ORDER.length; i++) {
      total += cell[i] * runValueTable[CLASS_ORDER[i]];
    }
    return total;
  }

  function initWhatIf(section, grid) {
    var evSlider = qs('[data-role="whatif-ev-slider"]', section);
    var laSlider = qs('[data-role="whatif-la-slider"]', section);
    var evValueEl = qs('[data-role="whatif-ev-value"]', section);
    var laValueEl = qs('[data-role="whatif-la-value"]', section);
    var fixedContextEl = qs('[data-role="whatif-fixed-context"]', section);
    var resetButton = qs('[data-role="whatif-reset"]', section);
    var probBarsEl = qs('[data-role="whatif-prob-bars"]', section);
    var actualExpectedEl = qs('[data-role="whatif-actual-expected-rv"]', section);
    var modifiedExpectedEl = qs('[data-role="whatif-modified-expected-rv"]', section);
    var changeExpectedEl = qs('[data-role="whatif-change-expected-rv"]', section);

    var runValueTable = grid.model_configuration.run_value_table;
    var state = {
      evIndex: grid.original_grid_index.ev_index,
      laIndex: grid.original_grid_index.la_index,
    };

    function currentCell() {
      var rowIndex = state.evIndex * grid.launch_angle_values.length + state.laIndex;
      return grid.grid[rowIndex];
    }

    function render() {
      var evValue = grid.exit_velocity_values[state.evIndex];
      var laValue = grid.launch_angle_values[state.laIndex];

      evSlider.max = String(grid.exit_velocity_values.length - 1);
      evSlider.value = String(state.evIndex);
      evSlider.setAttribute("aria-valuetext", evValue + " mph");
      evValueEl.textContent = evValue + " mph";

      laSlider.max = String(grid.launch_angle_values.length - 1);
      laSlider.value = String(state.laIndex);
      laSlider.setAttribute("aria-valuetext", laValue + "°");
      laValueEl.textContent = laValue + "°";

      var cell = currentCell();
      probBarsEl.innerHTML = "";
      CLASS_ORDER.forEach(function (cls, idx) {
        var pct = cell[idx] * 100;

        var row = document.createElement("div");
        row.className = "demo-prob-row";

        var label = document.createElement("span");
        label.className = "demo-prob-label";
        label.textContent = CLASS_LABELS[cls];
        row.appendChild(label);

        var track = document.createElement("span");
        track.className = "demo-prob-bar-track";
        var fill = document.createElement("span");
        fill.className = "demo-prob-bar-fill simulator-prob-bar-fill";
        fill.style.width = pct.toFixed(2) + "%";
        track.appendChild(fill);
        row.appendChild(track);

        var pctEl = document.createElement("span");
        pctEl.className = "demo-prob-pct";
        pctEl.textContent = pct.toFixed(1) + "%";
        row.appendChild(pctEl);

        probBarsEl.appendChild(row);
      });

      var actualExpected = grid.original_expected_run_value;
      var modifiedExpected = expectedRunValue(cell, runValueTable);
      actualExpectedEl.textContent = formatSigned(actualExpected);
      modifiedExpectedEl.textContent = formatSigned(modifiedExpected);
      changeExpectedEl.textContent = formatSigned(modifiedExpected - actualExpected);

      var fixed = grid.fixed_context;
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
    }

    evSlider.addEventListener("input", function () {
      state.evIndex = parseInt(evSlider.value, 10);
      render();
    });
    laSlider.addEventListener("input", function () {
      state.laIndex = parseInt(laSlider.value, 10);
      render();
    });
    resetButton.addEventListener("click", function () {
      state.evIndex = grid.original_grid_index.ev_index;
      state.laIndex = grid.original_grid_index.la_index;
      render();
    });

    render();
    section.hidden = false;
  }

  function initWhatIfSection() {
    var section = qs('[data-role="play-whatif-section"]');
    if (!section) return;

    var params = new URLSearchParams(window.location.search);
    var playId = params.get("id");
    if (!playId) return;
    // Phase 8: the play-id contract lives once, in app.js, alongside the
    // signed-number primitive this file already shares from there.
    var validated = window.ContactLuck && window.ContactLuck.gamePkFromPlayId
      ? window.ContactLuck.gamePkFromPlayId(playId)
      : (/^(\d+)-\d+-\d+$/.exec(playId) || [])[1] || null;
    if (!validated) return; // Malformed id -- never even attempt a fetch.

    fetch("/explore/showcase-sensitivity/" + encodeURIComponent(playId) + ".json")
      .then(function (response) {
        if (!response.ok) throw new Error("sensitivity grid not available: " + response.status);
        return response.json();
      })
      .then(function (grid) {
        initWhatIf(section, grid);
      })
      .catch(function () {
        // No sensitivity grid for this play (not interactive-eligible, or
        // a genuine fetch failure) -- leave the section hidden. Never show
        // a disabled/broken simulator.
      });
  }

  document.addEventListener("DOMContentLoaded", initWhatIfSection);
})();
