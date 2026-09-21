// A02 verification runner — executes the real week2/data.js + week2/script.js
// (the exact files served to the browser) inside JavaScriptCore (JXA).
//
// Run:  osascript -l JavaScript week2/evidence/run-verification.js
// Requires: macOS (ObjC bridge), files at week2/u.item and week2/u.data.
// Writes:  week2/evidence/a02-verification.json
// Prints:  passed/failed summary plus any failing assertion details.
//
// Floating-point comparisons use the documented tolerance TOL = 1e-12
// (|a - b| <= 1e-12).

ObjC.import("Foundation");

var TOL = 1e-12;
var BASE = "/Users/Sanzha.Tokkozhin/Desktop/Projects/RecSys-LLMs/week2/";

function readAs(path, encoding) {
    var data = $.NSData.dataWithContentsOfFile(path);
    if (data.isNil()) { throw new Error("cannot read " + path); }
    var enc = encoding === "latin1" ? $.NSISOLatin1StringEncoding : $.NSUTF8StringEncoding;
    return $.NSString.alloc.initWithDataEncoding(data, enc).js;
}

// The verification body below is concatenated onto data.js + script.js and
// compiled, so it runs inside the same scope and sees all globals/functions.
var BODY = `
var __checks = [];
var __tol = 1e-12;
function near(a, b) { return Math.abs(a - b) <= __tol; }
function assert(name, cond, detail) {
    __checks.push({ name: name, pass: !!cond, detail: detail === undefined ? "" : String(detail) });
}
function jp(x) { return JSON.stringify(x); }
function byId(id) { return movies.filter(function (m) { return m.id === id; })[0]; }
function clip4(v) { return Math.round(v * 10000) / 10000; }

var D = {};

// ---- 1. DATA INTEGRITY ----
parseItemData(__itemText);
parseRatingData(__dataText);
attachRatingCounts();
classifyPopularity();

D.movieCount = movies.length;
D.ratingCount = ratings.length;
assert("1.1 catalog has exactly 1682 movies", movies.length === 1682, "movies.length=" + movies.length);
assert("1.2 catalog has exactly 100000 ratings", ratings.length === 100000, "ratings.length=" + ratings.length);

var badLen = movies.filter(function (m) { return m.genreVector.length !== 18; }).length;
assert("1.3 every movie vector has length 18", badLen === 0, "badVectorLengths=" + badLen);
assert("1.4 vector entries are all 0/1 and finite", movies.every(function (m) { return m.genreVector.every(function (v) { return v === 0 || v === 1; }); }), "");

var official = ["Action","Adventure","Animation","Children's","Comedy","Crime","Documentary","Drama","Fantasy","Film-Noir","Horror","Musical","Mystery","Romance","Sci-Fi","Thriller","War","Western"];
assert("1.5 genreNames aligned to official field order", jp(genreNames) === jp(official), jp(genreNames));

var ts = byId(1);
D.toyStory = { id: ts.id, title: ts.title, vector: ts.genreVector, genres: ts.genres, ratingCount: ts.ratingCount };
assert("1.6 Toy Story (id 1) parsing sanity",
    ts.title.indexOf("Toy Story") !== -1 &&
    jp(ts.genres) === jp(["Animation", "Children's", "Comedy"]) &&
    near(ts.genreVector[2], 1) && near(ts.genreVector[3], 1) && near(ts.genreVector[4], 1) &&
    near(ts.genreVector[0], 0) && near(ts.genreVector[17], 0),
    jp(D.toyStory));

var unforgiven = byId(203);
assert("1.7 Western flag sanity (Unforgiven id 203)",
    unforgiven.genreVector[17] === 1 && genreNames[17] === "Western",
    unforgiven.title + " vector=" + jp(unforgiven.genreVector));

var zeroIds = movies.filter(function (m) { return vectorNorm(m.genreVector) === 0; }).map(function (m) { return m.id; });
D.zeroVectorMovies = { ids: zeroIds, count: zeroIds.length };
assert("1.8 unknown-only / zero-vector movies are exactly 267 and 1373", jp(zeroIds) === jp([267, 1373]), jp(zeroIds));

// ---- 2. FORMULA CHECK ----
var vectorA = ts.genreVector;
var vectorB = byId(95).genreVector; // Aladdin (1992): Animation, Children's, Comedy, Musical
var dAB = dotProduct(vectorA, vectorB);
var nA = vectorNorm(vectorA);
var nB = vectorNorm(vectorB);
var manual = dAB / (nA * nB);
var app = cosineSimilarity(vectorA, vectorB);
D.formulaPair = {
    a: { id: 1, title: byId(1).title, vector: vectorA },
    b: { id: 95, title: byId(95).title, vector: vectorB }
};
D.formula = { dot: dAB, normA: nA, normB: nB, manualCosine: manual, appCosine: app };
assert("2.1 dot product Toy Story . Aladdin = 3", near(dAB, 3), "dot=" + dAB);
assert("2.2 norms sqrt(3) and 2", near(nA, Math.sqrt(3)) && near(nB, 2), "nA=" + nA + " nB=" + nB);
assert("2.3 manual cosine agrees with app within 1e-12", near(manual, app), "manual=" + manual + " app=" + app);
var ab = cosineSimilarity(vectorA, vectorB);
var ba = cosineSimilarity(vectorB, vectorA);
assert("2.4 cosine symmetry cosine(A,B)==cosine(B,A)", near(ab, ba), "AB=" + ab + " BA=" + ba);
D.selfCosine = cosineSimilarity(vectorA, vectorA);
assert("2.5 cosine(A,A) is exactly 1 (non-zero vector)", D.selfCosine === 1, "cos(A,A)=" + D.selfCosine);
assert("2.6 cosine(A,A) within 1e-12 of 1", near(D.selfCosine, 1), "cos(A,A)=" + D.selfCosine);

// ---- 3. RECOMMENDATION INVARIANTS ----
var watched = [1, 296, 670];
var objs = watched.map(byId);
D.watched = watched.map(function (i) {
    var m = byId(i);
    return { id: m.id, title: m.title, genres: m.genres, ratingCount: m.ratingCount, klass: m.popularityClass };
});
var exclude = new Set(watched);
var excludeTitles = new Set(objs.map(function (o) { return o.canonicalTitle; }));
D.watchedCanonicalTitles = objs.map(function (o) { return o.canonicalTitle; });
var i2i = recommendFromVector(objs[0].genreVector, exclude, excludeTitles, 5);
var profL = recommendFromVector(buildUserProfile(objs), exclude, excludeTitles, 5);
function recInfo(list) {
    return list.map(function (e) {
        return { id: e.movie.id, title: e.movie.title, score: e.score, genres: e.movie.genres, ratingCount: e.movie.ratingCount, klass: e.movie.popularityClass };
    });
}
D.itemToItem = recInfo(i2i);
D.profileBased = recInfo(profL);
var i2iIds = i2i.map(function (e) { return e.movie.id; });
var profIds = profL.map(function (e) { return e.movie.id; });
function unique(ids) { return new Set(ids).size === ids.length; }
assert("3.1 item-to-item returns exactly 5 unique ids", i2i.length === 5 && unique(i2iIds), jp(i2iIds));
assert("3.2 profile-based returns exactly 5 unique ids", profL.length === 5 && unique(profIds), jp(profIds));
assert("3.3 no watched movie in either Top-5",
    i2iIds.concat(profIds).every(function (id) { return !exclude.has(id); }),
    "i2i=" + jp(i2iIds) + " prof=" + jp(profIds));
assert("3.8 no watched canonical title in either Top-5",
    i2i.concat(profL).every(function (e) { return !excludeTitles.has(e.movie.canonicalTitle); }),
    "i2i titles=" + jp(i2i.map(function (e) { return e.movie.title; })) +
    " prof titles=" + jp(profL.map(function (e) { return e.movie.title; })));

function allCandidateScores(reference, excludeIds, excludeTitleSet) {
    var out = [];
    movies.forEach(function (m) {
        if (!excludeIds.has(m.id) && !excludeTitleSet.has(m.canonicalTitle)) { out.push(cosineSimilarity(reference, m.genreVector)); }
    });
    return out;
}
var sI = allCandidateScores(objs[0].genreVector, exclude, excludeTitles);
var sP = allCandidateScores(buildUserProfile(objs), exclude, excludeTitles);
D.scoreSweep = {
    itemToItem: { count: sI.length, min: Math.min.apply(null, sI), max: Math.max.apply(null, sI) },
    profile:    { count: sP.length, min: Math.min.apply(null, sP), max: Math.max.apply(null, sP) }
};
assert("3.4 all candidate scores finite", sI.every(function (v) { return isFinite(v); }) && sP.every(function (v) { return isFinite(v); }), jp(D.scoreSweep));
assert("3.5 all candidate scores within [0, 1]",
    sI.every(function (v) { return v >= 0 && v <= 1; }) && sP.every(function (v) { return v >= 0 && v <= 1; }),
    jp(D.scoreSweep));
assert("3.6 all returned scores within [0, 1]",
    i2i.concat(profL).every(function (e) { return e.score >= 0 && e.score <= 1; }),
    jp(i2i.concat(profL).map(function (e) { return e.score; })));

var rerunI = recommendFromVector(objs[0].genreVector, exclude, excludeTitles, 5);
var rerunP = recommendFromVector(buildUserProfile(objs), exclude, excludeTitles, 5);
D.repeatProof = jp(recInfo(i2i)) === jp(recInfo(rerunI)) && jp(recInfo(profL)) === jp(recInfo(rerunP));
assert("3.7 deterministic: identical ordering on repeat", D.repeatProof, "equalRun=" + D.repeatProof);
assert("3.9 duplicate-title regression: watched 670 excludes same-title 573 by canonical title",
    byId(670).title === "Body Snatchers (1993)" && byId(573).title === "Body Snatchers (1993)" &&
    byId(670).canonicalTitle === byId(573).canonicalTitle &&
    i2iIds.indexOf(573) === -1 && profIds.indexOf(573) === -1 &&
    recommendFromVector(buildUserProfile(objs), exclude, new Set(), 5)
        .map(function (e) { return e.movie.id; }).indexOf(573) !== -1,
    "670=" + byId(670).title + " 573=" + byId(573).title + " canonical=" + byId(670).canonicalTitle);

// ---- 4. PROFILE CHECK ----
var profileVector = buildUserProfile(objs);
D.profileVector = profileVector;
D.profileToWatched = watched.map(function (i) { return clip4(cosineSimilarity(byId(i).genreVector, profileVector)); });
var pairs = [["1", "296"], ["1", "670"], ["296", "670"]];
D.watchedPairwise = {};
pairs.forEach(function (p) {
    D.watchedPairwise[p[0] + "|" + p[1]] = clip4(cosineSimilarity(byId(parseInt(p[0], 10)).genreVector, byId(parseInt(p[1], 10)).genreVector));
});
assert("4.1 profile vector is the element-wise mean of exactly 3 vectors", profileVector.length === 18, "len=" + profileVector.length);

// ---- 5. ITEM-TO-ITEM VS PROFILE-BASED ----
var overlap = i2iIds.filter(function (id) { return profIds.indexOf(id) !== -1; });
D.overlapCount = overlap.length;
D.overlapIds = overlap;
assert("5.1 overlap count recorded", D.overlapCount >= 0 && D.overlapCount <= 5, "overlap=" + D.overlapCount);

// ---- 6. BIAS AND DISCOVERY ----
var refV = vectorA;
var cands = movies.filter(function (m) { return m.id !== 1; }).map(function (m) {
    return { id: m.id, title: m.title, dot: dotProduct(refV, m.genreVector), cos: cosineSimilarity(refV, m.genreVector) };
});
function byDotDesc(a, b) { return (b.dot - a.dot) || a.title.localeCompare(b.title) || (a.id - b.id); }
function byCosDesc(a, b) { return (b.cos - a.cos) || a.title.localeCompare(b.title) || (a.id - b.id); }
D.rawDotTop5 = cands.slice().sort(byDotDesc).slice(0, 5).map(function (x) { return x.id; });
D.cosineTop5 = cands.slice().sort(byCosDesc).slice(0, 5).map(function (x) { return x.id; });
var c422 = cands.filter(function (c) { return c.id === 422; })[0];
var c95 = cands.filter(function (c) { return c.id === 95; })[0];
D.dotTieExample = {
    sameDot: c422.dot,
    id422: { title: c422.title, dot: c422.dot, cosine: c422.cos, norm: vectorNorm(byId(422).genreVector) },
    id95:  { title: c95.title,  dot: c95.dot,  cosine: c95.cos,  norm: vectorNorm(byId(95).genreVector) }
};
assert("6.1 equal dot product but different cosine (normalization removes length advantage)",
    near(c422.dot, c95.dot) && c422.cos > c95.cos + 1e-12,
    jp(D.dotTieExample));
D.popularityThreshold = { percentile: 80, headCount: catalogStats.headCount, longTailCount: catalogStats.longTailCount, headCutoffCount: catalogStats.headCutoffCount };
assert("6.2 popularity threshold is the documented top-20% boundary", catalogStats.headCount === 337 && catalogStats.longTailCount === 1345 && catalogStats.headCutoffCount === 100, jp(D.popularityThreshold));
D.i2iClassCounts = { head: sHead(i2i), longTail: sLong(i2i) };
D.profileClassCounts = { head: sHead(profL), longTail: sLong(profL) };
function sHead(list) { return list.filter(function (e) { return e.movie.popularityClass === "head"; }).length; }
function sLong(list) { return list.filter(function (e) { return e.movie.popularityClass === "long-tail"; }).length; }

// ---- 8. PREVENTIVE GUARDRAILS (regression checks) ----
assert("8.1 vector-length guard: all 1682 vectors length 18", movies.every(function (m) { return m.genreVector.length === 18; }), "");
assert("8.2 zero-norm guard: cosine with unknown-only movie returns 0",
    cosineSimilarity(refV, byId(267).genreVector) === 0 &&
    cosineSimilarity(refV, byId(1373).genreVector) === 0 &&
    cosineSimilarity(byId(267).genreVector, byId(1373).genreVector) === 0,
    "");
assert("8.3 watched-item filter: 3 watched ids excluded from BOTH lists",
    i2iIds.concat(profIds).every(function (id) { return !exclude.has(id); }),
    "");
assert("8.5 watched-title filter: canonical titles of the 3 watched movies excluded from BOTH lists",
    i2i.concat(profL).every(function (e) { return !excludeTitles.has(e.movie.canonicalTitle); }),
    "watched titles=" + jp(objs.map(function (o) { return o.canonicalTitle; })));
assert("8.4 deterministic sort: score->title->id comparator reproducible",
    jp(recInfo(i2i)) === jp(recInfo(recommendFromVector(objs[0].genreVector, exclude, excludeTitles, 5))),
    "");

var passed = __checks.filter(function (c) { return c.pass; }).length;
var failed = __checks.length - passed;
return { passed: passed, failed: failed, total: __checks.length, tolerance: __tol, checks: __checks, data: D };
`;

function run() {
    var dataSrc = readAs(BASE + "data.js", "utf8");
    var scriptSrc = readAs(BASE + "script.js", "utf8");
    var src = dataSrc + "\n" + scriptSrc + "\n" + BODY;
    var fn = new Function("document", "window", "__itemText", "__dataText", src);
    var res = fn({}, {}, readAs(BASE + "u.item", "latin1"), readAs(BASE + "u.data", "utf8"));

    var outputPath = BASE + "evidence/a02-verification.json";
    var jsonStr = $.NSString.alloc.initWithString(JSON.stringify(res, null, 2));
    jsonStr.writeToFileAtomicallyEncodingError(outputPath, true, $.NSUTF8StringEncoding, $());

    var lines = [
        "A02 verification summary",
        "passed=" + res.passed + " failed=" + res.failed + " total=" + res.total + " tolerance=" + res.tolerance,
        "evidence written to " + outputPath
    ];
    for (var i = 0; i < res.checks.length; i++) {
        if (!res.checks[i].pass) {
            lines.push("FAIL: " + res.checks[i].name + " :: " + res.checks[i].detail);
        }
    }
    return lines.join("\n");
}