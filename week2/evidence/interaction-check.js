// Interaction logic check — executes the real week2/script.js (with DOM stubs)
// to verify the code paths behind: duplicate-selection validation, the
// zero-genre validation, both Top-5 renderers, and the comparison summary.
//
// Run: osascript -l JavaScript week2/evidence/interaction-check.js
// This complements the real headless-Chrome load check (see 07-browser-check.md);
// it is not a substitute for a visual/responsive browser pass.

ObjC.import("Foundation");

var BASE = "/Users/Sanzha.Tokkozhin/Desktop/Projects/RecSys-LLMs/week2/";

function readAs(path, encoding) {
    var data = $.NSData.dataWithContentsOfFile(path);
    if (data.isNil()) { throw new Error("cannot read " + path); }
    var enc = encoding === "latin1" ? $.NSISOLatin1StringEncoding : $.NSUTF8StringEncoding;
    return $.NSString.alloc.initWithDataEncoding(data, enc).js;
}

var RUN = `
var console = { log: function () {}, warn: function () {}, error: function () {} };
parseItemData(__itemText);
parseRatingData(__dataText);
attachRatingCounts();
classifyPopularity();

function makeEl(tag) {
    var el = {
        tag: tag,
        _text: "",
        _children: [],
        value: "",
        className: "",
        options: [{ selected: true, disabled: true, value: "" }],
        classList: {
            _s: [],
            add: function (c) { if (this._s.indexOf(c) === -1) { this._s.push(c); } },
            remove: function (c) { var i = this._s.indexOf(c); if (i !== -1) { this._s.splice(i, 1); } },
            contains: function (c) { return this._s.indexOf(c) !== -1; }
        },
        addEventListener: function () {},
        remove: function () {},
        appendChild: function (c) { this._children.push(c); return c; }
    };
    Object.defineProperty(el, "textContent", {
        get: function () { return this._text; },
        set: function (v) { this._text = String(v); }
    });
    return el;
}

var els = {};
["movie-select-1", "movie-select-2", "movie-select-3", "result", "recommend-btn",
 "panels", "summary-box", "item-to-item-list", "profile-list",
 "panel-i2i-title", "panel-prof-title", "summary"].forEach(function (id) {
    els[id] = makeEl(id);
});

var document = {
    getElementById: function (id) { return els[id] || (els[id] = makeEl(id)); },
    createElement: function (tag) { return makeEl(tag); },
    createTextNode: function (v) { var n = makeEl("text"); n._text = String(v); return n; }
};
var window = {};

function titleOf(li) {
    var parts = [];
    function walk(n) {
        if (n._children && n._children.length) { n._children.forEach(walk); }
        else if (n._text) { parts.push(n._text); }
    }
    walk(li);
    return parts.join(" / ");
}

// ---- scenario A: duplicate selection -> validation error ----
els["movie-select-1"].value = "1";
els["movie-select-2"].value = "1";
els["movie-select-3"].value = "670";
getRecommendations();
var dupMessage = els["result"].textContent;

// ---- scenario B: valid distinct triple -> two lists + summary ----
els["movie-select-1"].value = "1";
els["movie-select-2"].value = "296";
els["movie-select-3"].value = "670";
getRecommendations();
var i2iItems = els["item-to-item-list"]._children;
var profItems = els["profile-list"]._children;
var summaryText = els["summary"]._children.map(function (p) {
    return p._children.map(function (n) { return n._text; }).join("");
}).join(String.fromCharCode(10));

// ---- scenario C: first movie has no recognized genres -> validation ----
els["movie-select-1"].value = "267";
els["movie-select-2"].value = "296";
els["movie-select-3"].value = "670";
getRecommendations();
var zeroGenreMessage = els["result"].textContent;

var i2iTitle = els["panel-i2i-title"].textContent;

return {
    duplicateMessage: dupMessage,
    i2iCount: i2iItems.length,
    profileCount: profItems.length,
    i2iTitles: i2iItems.map(titleOf),
    profileTitles: profItems.map(titleOf),
    panelI2iHeading: i2iTitle,
    summaryText: summaryText,
    panelsHiddenStill: els["panels"].classList.contains("hidden"),
    zeroGenreMessage: zeroGenreMessage
};
`;

function run() {
    var dataSrc = readAs(BASE + "data.js", "utf8");
    var scriptSrc = readAs(BASE + "script.js", "utf8");
    var src = dataSrc + "\n" + scriptSrc + "\n" + RUN;
    var fn = new Function("document", "window", "__itemText", "__dataText", src);
    return JSON.stringify(fn({}, {}, readAs(BASE + "u.item", "latin1"), readAs(BASE + "u.data", "utf8")));
}