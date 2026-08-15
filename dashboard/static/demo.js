// Contact Luck v1.3.0 -- "/demo/" page stage reveal + illustrative ball
// animation. Purely presentational: reads values already rendered into the
// page (server-computed probability-bar widths via a CSS custom property,
// the field diagram's own start/control/end coordinates) and never
// fetches, recomputes, or requests a score from anywhere.
//
// The ball's motion is a small `requestAnimationFrame` loop interpolating
// the circle's `cx`/`cy` along a quadratic Bezier curve -- NOT SVG SMIL
// (`<animateMotion>`). An earlier version used `<animateMotion>`; real
// -browser verification (position-sampled over time, including through a
// second Replay) showed two problems: its motion is applied as a transform
// that never touches `cx`/`cy` (so the element's rendered position can't be
// read or asserted from its own attributes), and restarting an already
// -completed (`fill="freeze"`) SMIL animation via `beginElement()` is a
// well-documented cross-browser/version reliability gap. Driving `cx`/`cy`
// directly with `requestAnimationFrame` removes both problems: the ball's
// position is always exactly what its attributes say, and Replay is just
// "cancel any in-flight frame, reset attributes, start a fresh loop."
(function () {
  "use strict";

  var STAGE_DELAY_MS = 650;
  var STAGE_COUNT = 4;
  var REALITY_STAGE = 3;
  var AUTOPLAY_DELAY_MS = 300;
  // Matches the existing stage pacing: Stage 3 ("Reality") reveals at
  // STAGE_DELAY_MS * 2 = 1300ms, so the ball's illustrative flight also
  // takes 1300ms -- it lands right as the actual result is revealed.
  var BALL_TRAVEL_MS = 1300;

  function prefersReducedMotion() {
    return (
      window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches
    );
  }

  function easeOutCubic(t) {
    var inv = 1 - t;
    return 1 - inv * inv * inv;
  }

  function quadraticBezier(t, p0, p1, p2) {
    var inv = 1 - t;
    return {
      x: inv * inv * p0.x + 2 * inv * t * p1.x + t * t * p2.x,
      y: inv * inv * p0.y + 2 * inv * t * p1.y + t * t * p2.y,
    };
  }

  function ballCurvePoints(ball) {
    return {
      start: { x: Number(ball.getAttribute("data-start-x")), y: Number(ball.getAttribute("data-start-y")) },
      control: { x: Number(ball.getAttribute("data-control-x")), y: Number(ball.getAttribute("data-control-y")) },
      end: { x: Number(ball.getAttribute("data-end-x")), y: Number(ball.getAttribute("data-end-y")) },
    };
  }

  function setBallPosition(ball, point) {
    ball.setAttribute("cx", point.x.toFixed(2));
    ball.setAttribute("cy", point.y.toFixed(2));
  }

  function cancelBallAnimation(ball) {
    if (ball._demoRafId != null) {
      window.cancelAnimationFrame(ball._demoRafId);
      ball._demoRafId = null;
    }
  }

  function animateBall(ball) {
    var points = ballCurvePoints(ball);
    var startTime = null;

    function frame(now) {
      if (startTime === null) startTime = now;
      var t = Math.min((now - startTime) / BALL_TRAVEL_MS, 1);
      setBallPosition(ball, quadraticBezier(easeOutCubic(t), points.start, points.control, points.end));
      if (t < 1) {
        ball._demoRafId = window.requestAnimationFrame(frame);
      } else {
        ball._demoRafId = null;
      }
    }

    ball._demoRafId = window.requestAnimationFrame(frame);
  }

  function resetCard(card) {
    var stages = card.querySelectorAll(".demo-stage");
    Array.prototype.forEach.call(stages, function (stage) {
      stage.classList.remove("demo-stage-revealed");
    });
    // The observed-outcome highlight in the Stage-2 probability bars is
    // gated on this class (see static/style.css) so it appears alongside
    // the Stage-3 "Reality" reveal, not earlier -- reset it here so a
    // replay doesn't start with the answer already highlighted.
    card.classList.remove("demo-reality-revealed");

    var ball = card.querySelector(".demo-ball");
    if (ball) {
      cancelBallAnimation(ball);
      var points = ballCurvePoints(ball);
      setBallPosition(ball, points.start);
    }
  }

  function revealStage(card, stageNumber) {
    var stage = card.querySelector('.demo-stage[data-stage="' + stageNumber + '"]');
    if (stage) stage.classList.add("demo-stage-revealed");
    if (stageNumber === REALITY_STAGE) card.classList.add("demo-reality-revealed");
  }

  function playCard(card) {
    resetCard(card);
    var ball = card.querySelector(".demo-ball");

    if (prefersReducedMotion()) {
      // No timed reveal, no animated ball motion -- show the final state
      // immediately, with the ball placed directly at its (illustrative)
      // endpoint.
      for (var s = 1; s <= STAGE_COUNT; s++) revealStage(card, s);
      if (ball) setBallPosition(ball, ballCurvePoints(ball).end);
      return;
    }

    if (ball) animateBall(ball);

    for (var i = 1; i <= STAGE_COUNT; i++) {
      window.setTimeout(
        (function (stageNumber) {
          return function () {
            revealStage(card, stageNumber);
          };
        })(i),
        STAGE_DELAY_MS * (i - 1)
      );
    }
  }

  function initDemoPage() {
    var cards = document.querySelectorAll(".demo-card");
    if (!cards.length) return;

    var playButton = document.querySelector('[data-role="demo-play-all"]');
    var hasPlayedOnce = false;

    function playAll() {
      Array.prototype.forEach.call(cards, playCard);
      if (playButton && !hasPlayedOnce) {
        hasPlayedOnce = true;
        playButton.textContent = "Replay";
      }
    }

    if (playButton) {
      playButton.addEventListener("click", playAll);
    }

    window.setTimeout(playAll, prefersReducedMotion() ? 0 : AUTOPLAY_DELAY_MS);
  }

  document.addEventListener("DOMContentLoaded", initDemoPage);
})();
