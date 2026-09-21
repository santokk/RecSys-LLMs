const WATCHED_COUNT = 3;
const RECOMMENDATION_COUNT = 5;
const SELECTOR_IDS = ['movie-select-1', 'movie-select-2', 'movie-select-3'];

// Initialize the application when the window loads
window.onload = async function() {
    try {
        const resultElement = document.getElementById('result');
        resultElement.textContent = "Loading movie data...";
        resultElement.className = 'loading';

        await loadData();

        populateMoviesDropdowns();
        document.getElementById('recommend-btn').addEventListener('click', getRecommendations);

        resultElement.textContent = `Data loaded. ${movies.length} movies, ${ratings.length} ratings. Select three movies and press "Get Recommendations".`;
        resultElement.className = 'success';
    } catch (error) {
        console.error('Initialization error:', error);
        // Error message already set in data.js
    }
};

// Dot product of two equal-length numeric vectors
function dotProduct(vectorA, vectorB) {
    let sum = 0;
    for (let i = 0; i < vectorA.length; i++) {
        sum += vectorA[i] * vectorB[i];
    }
    return sum;
}

// Euclidean (L2) norm of a numeric vector
function vectorNorm(vector) {
    return Math.sqrt(dotProduct(vector, vector));
}

// Cosine similarity: dot(A, B) / (norm(A) * norm(B)).
// A zero-magnitude vector has undefined cosine; return 0 so a movie with no
// recognized genres never produces NaN or Infinity. The result is clamped to
// the mathematically valid [-1, 1] so floating-point rounding cannot push
// self-similarity above 1.
function cosineSimilarity(vectorA, vectorB) {
    const normA = vectorNorm(vectorA);
    const normB = vectorNorm(vectorB);
    if (normA === 0 || normB === 0) return 0;
    const value = dotProduct(vectorA, vectorB) / (normA * normB);
    return Math.max(-1, Math.min(1, value));
}

// Element-wise mean of exactly three movie genre vectors -> user profile.
function buildUserProfile(watchedMovies) {
    if (watchedMovies.length !== WATCHED_COUNT) {
        throw new Error(`Expected exactly ${WATCHED_COUNT} watched movies, got ${watchedMovies.length}`);
    }
    const profile = new Array(GENRE_VECTOR_SIZE).fill(0);
    for (const movie of watchedMovies) {
        for (let i = 0; i < GENRE_VECTOR_SIZE; i++) {
            profile[i] += movie.genreVector[i];
        }
    }
    for (let i = 0; i < GENRE_VECTOR_SIZE; i++) {
        profile[i] /= WATCHED_COUNT;
    }
    return profile;
}

// Deterministic sort: score descending, then title ascending, then id
// ascending so two same-scored same-titled movies still have a stable order.
function compareCandidates(a, b) {
    if (a.score !== b.score) return b.score - a.score;
    const byTitle = a.movie.title.localeCompare(b.movie.title);
    if (byTitle !== 0) return byTitle;
    return a.movie.id - b.movie.id;
}

// Score every movie that is not watched against referenceVector, then return
// the top `limit` by the deterministic order. A candidate is skipped when its
// ID is watched OR its canonical title matches a watched movie, so duplicate
// catalog rows for the same title cannot leak a watched movie into results.
function recommendFromVector(referenceVector, excludeIds, excludeTitles, limit) {
    const scored = [];
    for (const movie of movies) {
        if (excludeIds.has(movie.id)) continue;
        if (excludeTitles.has(movie.canonicalTitle)) continue;
        scored.push({ movie, score: cosineSimilarity(referenceVector, movie.genreVector) });
    }
    scored.sort(compareCandidates);
    return scored.slice(0, limit);
}

// Populate all three selects with the same alphabetically sorted movies
function populateMoviesDropdowns() {
    const sortedMovies = [...movies].sort((a, b) => a.title.localeCompare(b.title));

    for (const id of SELECTOR_IDS) {
        const selectElement = document.getElementById(id);
        while (selectElement.options.length > 1) {
            selectElement.remove(1);
        }
        for (const movie of sortedMovies) {
            const option = document.createElement('option');
            option.value = movie.id;
            option.textContent = `${movie.title} (${movie.genres.length ? movie.genres.join(', ') : 'no genres'})`;
            selectElement.appendChild(option);
        }
    }
}

// Read the three selected movie objects
function readWatchedMovies() {
    return SELECTOR_IDS.map(id => movies.find(movie => movie.id === parseInt(document.getElementById(id).value, 10)));
}

// Main recommendation function
function getRecommendations() {
    const resultElement = document.getElementById('result');

    try {
        resultElement.textContent = "Calculating recommendations...";
        resultElement.className = 'loading';

        const movieObjects = readWatchedMovies();
        const missing = movieObjects.some(movie => !movie);
        if (missing) {
            setMessage("Please select three movies before generating recommendations.", 'error');
            return;
        }

        const duplicateLabels = [];
        const seen = new Map();
        for (const movie of movieObjects) {
            if (seen.has(movie.id)) {
                duplicateLabels.push(`"${movie.title}"`);
            }
            seen.set(movie.id, movie);
        }
        if (duplicateLabels.length > 0) {
            setMessage(`Please choose three distinct movies. Duplicate: ${duplicateLabels.join(', ')}.`, 'error');
            return;
        }

        const firstMovie = movieObjects[0];
        if (vectorNorm(firstMovie.genreVector) === 0) {
            setMessage(
                `"${firstMovie.title}" has no recognized genres, so item-to-item similarity cannot be calculated for it. ` +
                'Choose a different first movie, or keep it as the second or third selection.',
                'error'
            );
            return;
        }

        const excludeIds = new Set(movieObjects.map(movie => movie.id));
        const excludeTitles = new Set(movieObjects.map(movie => movie.canonicalTitle));

        const itemToItem = recommendFromVector(
            firstMovie.genreVector, excludeIds, excludeTitles, RECOMMENDATION_COUNT
        );

        const profileVector = buildUserProfile(movieObjects);
        const profileBased = recommendFromVector(
            profileVector, excludeIds, excludeTitles, RECOMMENDATION_COUNT
        );

        renderLists(firstMovie, itemToItem, profileBased);
        renderSummary(movieObjects, itemToItem, profileBased);

        resultElement.textContent = 'Recommendations ready.';
        resultElement.className = 'success';
        document.getElementById('panels').classList.remove('hidden');
        document.getElementById('summary-box').classList.remove('hidden');
    } catch (error) {
        console.error('Error in recommendation calculation:', error);
        setMessage('An unexpected error occurred while calculating recommendations.', 'error');
    }
}

function setMessage(text, className) {
    const resultElement = document.getElementById('result');
    resultElement.textContent = text;
    resultElement.className = className;
}

function formatScore(score) {
    return score.toFixed(3);
}

// Build one <li> for a single recommendation entry
function appendRecommendation(listElement, rank, entry) {
    const movie = entry.movie;
    const item = document.createElement('li');
    item.className = 'rec-item';

    const head = document.createElement('div');
    head.className = 'rec-head';

    const titleNode = document.createElement('span');
    titleNode.className = 'rec-title';
    titleNode.textContent = `${rank}. ${movie.title}`;

    const scoreNode = document.createElement('span');
    scoreNode.className = 'rec-score';
    scoreNode.textContent = `cosine ${formatScore(entry.score)}`;

    head.appendChild(titleNode);
    head.appendChild(scoreNode);

    const meta = document.createElement('div');
    meta.className = 'rec-meta';
    meta.textContent = `Genres: ${movie.genres.length ? movie.genres.join(', ') : '—'}`;

    const count = document.createElement('div');
    count.className = 'rec-count';
    count.textContent = `Ratings: ${movie.ratingCount}`;

    item.appendChild(head);
    item.appendChild(meta);
    item.appendChild(count);
    listElement.appendChild(item);
}

function renderLists(firstMovie, itemToItem, profileBased) {
    const i2iTitle = document.getElementById('panel-i2i-title');
    i2iTitle.textContent = `Item-to-Item Top-5 (based on "${firstMovie.title}")`;

    const i2iList = document.getElementById('item-to-item-list');
    i2iList.textContent = '';
    itemToItem.forEach((entry, index) => {
        appendRecommendation(i2iList, index + 1, entry);
    });

    const profileList = document.getElementById('profile-list');
    profileList.textContent = '';
    profileBased.forEach((entry, index) => {
        appendRecommendation(profileList, index + 1, entry);
    });
}

// Build the comparison summary with overlap, leakage, and long-tail counts.
function renderSummary(movieObjects, itemToItem, profileBased) {
    const watchedIds = new Set(movieObjects.map(movie => movie.id));
    const watchedTitles = new Set(movieObjects.map(movie => movie.canonicalTitle));
    const i2iIds = itemToItem.map(entry => entry.movie.id);
    const profileIds = profileBased.map(entry => entry.movie.id);
    const overlapIds = i2iIds.filter(id => profileIds.includes(id));

    const leaked = i2iIds.concat(profileIds).filter(id => {
        const movie = movieById(id);
        return watchedIds.has(id) || watchedTitles.has(movie.canonicalTitle);
    });
    const leakMessage = leaked.length === 0
        ? `None of the 10 recommendations is one of the ${WATCHED_COUNT} watched movies (checked by id and by title).`
        : `LEAKAGE FOUND: ${leaked.map(id => movieById(id).title).join(', ')}`;

    const summarize = (entries) => ({
        longTail: entries.filter(entry => entry.movie.popularityClass === 'long-tail').length,
        head: entries.filter(entry => entry.movie.popularityClass === 'head').length
    });
    const i2iStats = summarize(itemToItem);
    const profileStats = summarize(profileBased);

    const summaryElement = document.getElementById('summary');
    summaryElement.textContent = '';

    appendSummaryItem(summaryElement, 'Watched movies',
        movieObjects.map(movie => `"${movie.title}"`).join(', '));

    appendSummaryItem(summaryElement, 'Overlap between the two Top-5 lists',
        `${overlapIds.length} of 5${overlapIds.length ? `: ${overlapIds.map(id => movieById(id).title).join(', ')}` : ''}`);

    appendSummaryItem(summaryElement, 'Already-watched leakage', leakMessage);

    const header = (name, stats) =>
        `${name}: ${stats.longTail} lower-popularity (long-tail), ${stats.head} higher-popularity (head)`;
    appendSummaryItem(summaryElement, 'Popularity of recommendations',
        `${header('Item-to-Item', i2iStats)} · ${header('Profile', profileStats)}`);

    appendSummaryItem(summaryElement, 'How "lower-popularity" is defined',
        `head = top ${catalogStats.headCount} of ${movies.length} movies (20%) by rating count (tie-break: title, id); ` +
        `boundary = ${catalogStats.headCutoffCount} ratings. ` +
        `${catalogStats.longTailCount} movies fall in the long-tail. Classification is diagnostic only.`);
}

function appendSummaryItem(container, label, value) {
    const row = document.createElement('p');
    const labelNode = document.createElement('strong');
    labelNode.textContent = `${label}: `;
    row.appendChild(labelNode);
    row.appendChild(document.createTextNode(value));
    container.appendChild(row);
}

function movieById(id) {
    return movies.find(movie => movie.id === id);
}