var pageAnimation = null;
var body = null;
var introHeader = null;
var childElementCount = null;
var introHeaderChildren = null;
var hasResetSizing = true;

function IsAnimation(element) {
  if (element.tagName == "svg") {
    console.warn("you are attempting to modify the page Animation");
    return true;
  } else {
    return false;
  }
}

function DisableStaticChildElements(childElementCount, childElements) {
  for (var i = 0; i < childElementCount; i++) {
    // check we aren't modifying the animation
    if (!IsAnimation(childElements[i])) {
      // turn off the static version of the intro elements
      childElements[i].setAttribute("style", "display: none;");
    }
  }
}

function EnableStaticChildElements(childElementCount, childElements) {
  for (var i = 0; i < childElementCount; i++) {
    if (!IsAnimation(childElements[i])) {
      // clear the style i.e turns it back on by removing display : none
      childElements[i].removeAttribute("style");
    }
  }
}

function LoadAnimation(container) {
  var _container = introHeader.appendChild(document.createElement("div"));
  _container.setAttribute("id", "bodymovin");
  var data = {
    container: _container,
    renderer: "svg",
    loop: false,
    autoplay: true,
    path: "assets/data.json"
  };
  var _animation = bodymovin.loadAnimation(data);
  bodymovin.setQuality(2); // low quality shows no visual difference
  return _animation;
}

// with the following two functions, as the css styling values would come back in px rather than percentage,
// it's safe to just hard code the values that we are applying/resetting

function AdjustIntroElementsPositioning() {
  document
    .getElementById("name-arrow-wrapper")
    .setAttribute("style", "margin-top: -32%; left: 80.5%;");
  document
    .getElementById("code-arrow-wrapper")
    .setAttribute("style", "margin-top: -11.3%; left: 8%;");
  document
    .getElementById("intro-header-container")
    .setAttribute("style", "padding-bottom: 0;");
  document
    .getElementById("intro-icons-container")
    .setAttribute("style", "margin-top: 0%;");
  hasResetSizing = false;
}

function ResetIntroElementsPositioning() {
  document.getElementById("name-arrow-wrapper").removeAttribute("style");
  document.getElementById("code-arrow-wrapper").removeAttribute("style");
  document.getElementById("intro-header-container").removeAttribute("style");
  document.getElementById("intro-icons-container").removeAttribute("style");
  hasResetSizing = true;
}

function IsMobile() {
  if (window.outerWidth <= 545) {
    return true;
  } else {
    return false;
  }
}

function ApplyAnimationStyling(element) {
  // internet explorer svg sizing bug - fixed with our relevant css styling
  element.children[0].setAttribute("id", "bodymovin-svg");
}

document.addEventListener("DOMContentLoaded", function() {
  body = this.getElementsByTagName("body")[0];
  body.setAttribute("style", "opacity: 0;"); // hide the page to prevent FOUC
  introHeader = this.getElementById("intro-header");
  childElementCount = introHeader.childElementCount;
  introHeaderChildren = introHeader.children;
  if (!IsMobile()) {
    introHeader.style.background = "none"; // remove the static background image scribble
    // we let our animation play out
    pageAnimation = LoadAnimation(introHeader);
    // when the bodymovin data is loaded in
    pageAnimation.addEventListener("data_ready", function() {
      // go two levels down to get the animation svg container
      ApplyAnimationStyling(introHeader.lastChild);
    });
    pageAnimation.addEventListener("complete", function() {
      introHeader.lastChild.children[0].style.transform = null; // removes hardware acceleration we applied earlier
    });
    DisableStaticChildElements(childElementCount, introHeaderChildren);
    AdjustIntroElementsPositioning();
  }
});

// bind this to the onload event
function DisplayPage() {
  setCopyrightYear();
  body.removeAttribute("style"); // let the page display now that the animation is loaded
}
window.onload = DisplayPage;

window.onresize = function() {
  currentWindowWidth = this.outerWidth;
  // this check is for when the screen width starts off large than has shrunk down
  // the result is it allows us to switch to the static element & disable the animating one
  if (this.outerWidth <= 545) {
    if (!hasResetSizing) {
      ResetIntroElementsPositioning();
    }
    if (pageAnimation != null) {
      pageAnimation.destroy(); // if anim exists, remove and go to static element
      pageAnimation == null;
      document
        .getElementById("bodymovin")
        .setAttribute("style", "display: none;");
      EnableStaticChildElements(childElementCount, introHeaderChildren);
      introHeader.removeAttribute("style"); // enable the scribble
    }
  }
};

function setCopyrightYear() {
  document.getElementById("copyright").innerHTML =
    "© Ronan Quigley " + getYear();
}

function getYear() {
  return new Date().getFullYear();
}
