window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

// Ensure we subscribe to mkdocs-material navigation events only after
// MathJax and document$ are available. Poll briefly and then register the
// handler so typesetting runs after each client-side navigation without
// requiring a manual refresh.
(function subscribeWhenMathJaxReady() {
  function trySubscribe() {
    if (typeof document$ !== 'undefined' && typeof MathJax !== 'undefined') {
      try {
        document$.subscribe(() => {
          try {
            // Clear any previous typeset output cache (MathJax v3)
            if (MathJax && MathJax.startup && MathJax.startup.output && typeof MathJax.startup.output.clearCache === 'function') {
              MathJax.startup.output.clearCache();
            }

            // Clear typesetting state
            if (typeof MathJax.typesetClear === 'function') {
              MathJax.typesetClear();
            }

            // Reset TeX state (macros/environments)
            if (typeof MathJax.texReset === 'function') {
              MathJax.texReset();
            }

            // Re-typeset the current DOM fragment
            MathJax.typesetPromise();
          } catch (err) {
            // If clearing fails, still attempt a typeset as a fallback
            try { if (typeof MathJax.typesetPromise === 'function') MathJax.typesetPromise(); } catch (e) {}
            console.warn('MathJax navigation typeset error:', err);
          }
        });
      } catch (e) {
        // swallow errors; MathJax will usually typeset the initial page load.
        // If something goes wrong, avoid breaking page scripts.
      }
    } else {
      setTimeout(trySubscribe, 100);
    }
  }
  trySubscribe();
})();
