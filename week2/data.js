// Global variables for storing movie and rating data
let movies = [];
let ratings = [];

// Genre columns in u.item order: fields[5] through fields[23] (19 flags).
// The first flag, "unknown", is excluded from the recommendation vector.
const genreFlags = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
    "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western"
];

// The 18 meaningful genres used as the recommendation vector.
const genreNames = genreFlags.slice(1);
const GENRE_VECTOR_SIZE = genreNames.length;

// u.item uses extended ASCII (Latin-1); TextDecoder leaves the 0xE9 bytes in
// eight titles intact instead of replacing them like response.text() does.
const ITEM_ENCODING = 'iso-8859-1';

// Catalog-level popularity classification. Diagnostic only: rating counts are
// never used to rank recommendations.
const catalogStats = {
    headCount: 0,
    longTailCount: 0,
    headCutoffCount: 0,
    distribution: []
};

// Primary function to load data from files
async function loadData() {
    try {
        // Load and parse movie data
        const moviesResponse = await fetch('u.item');
        if (!moviesResponse.ok) {
            throw new Error(`Failed to load movie data: ${moviesResponse.status}`);
        }
        const moviesBuffer = await moviesResponse.arrayBuffer();
        const moviesText = new TextDecoder(ITEM_ENCODING).decode(moviesBuffer);
        const movieWarnings = parseItemData(moviesText);

        // Load and parse rating data
        const ratingsResponse = await fetch('u.data');
        if (!ratingsResponse.ok) {
            throw new Error(`Failed to load rating data: ${ratingsResponse.status}`);
        }
        const ratingsText = await ratingsResponse.text();
        const ratingWarnings = parseRatingData(ratingsText);

        attachRatingCounts();
        classifyPopularity();

        for (const warning of movieWarnings.concat(ratingWarnings)) {
            console.warn(warning);
        }
    } catch (error) {
        console.error('Error loading data:', error);
        const resultElement = document.getElementById('result');
        if (resultElement) {
            resultElement.textContent = `Error: ${error.message}. Please make sure u.item and u.data files are in the correct location.`;
            resultElement.className = 'error';
        }
        throw error; // Re-throw to allow script.js to handle the error
    }
}

// Parse movie data from u.item format
// Returns a list of warning strings for rows that could not be parsed.
function parseItemData(text) {
    const warnings = [];
    const lines = text.split('\n');

    for (let lineNo = 0; lineNo < lines.length; lineNo++) {
        const line = lines[lineNo];
        if (line.trim() === '') continue;

        const fields = line.split('|');
        if (fields.length !== 24) {
            warnings.push(`Malformed u.item row ${lineNo + 1}: expected 24 fields, got ${fields.length}`);
            continue;
        }

        const id = parseInt(fields[0], 10);
        const title = fields[1];
        if (!Number.isInteger(id) || title === '') {
            warnings.push(`Malformed u.item row ${lineNo + 1}: invalid id or empty title`);
            continue;
        }

        // The unknown flag lives at fields[5]; the 18 named genres span
        // fields[6] through fields[23], aligned with genreNames by index.
        const genreVector = genreNames.map((_, index) => (fields[6 + index] === '1' ? 1 : 0));
        const genres = genreNames.filter((_, index) => genreVector[index] === 1);

        movies.push({
            id,
            title,
            canonicalTitle: canonicalizeTitle(title),
            genres,
            genreVector,
            ratingCount: 0
        });
    }

    return warnings;
}

// Canonical title key used for title-level watched-item exclusion.
// Normalizes whitespace and case but preserves the year in the title, so
// remakes with different years stay distinct (e.g., duplicate catalog rows
// for "Body Snatchers (1993)" collide, while a 1994 remake would not).
function canonicalizeTitle(title) {
    return title.trim().toLowerCase().replace(/\s+/g, ' ');
}

// Parse rating data from u.data format
// Returns a list of warning strings for rows that could not be parsed.
function parseRatingData(text) {
    const warnings = [];
    const lines = text.split('\n');

    for (let lineNo = 0; lineNo < lines.length; lineNo++) {
        const line = lines[lineNo];
        if (line.trim() === '') continue;

        const fields = line.split('\t');
        if (fields.length !== 4) {
            warnings.push(`Malformed u.data row ${lineNo + 1}: expected 4 fields, got ${fields.length}`);
            continue;
        }

        const userId = parseInt(fields[0], 10);
        const itemId = parseInt(fields[1], 10);
        const rating = parseFloat(fields[2]);
        const timestamp = parseInt(fields[3], 10);
        if (!Number.isInteger(userId) || !Number.isInteger(itemId) ||
            Number.isNaN(rating) || !Number.isInteger(timestamp)) {
            warnings.push(`Malformed u.data row ${lineNo + 1}: non-numeric field`);
            continue;
        }

        ratings.push({ userId, itemId, rating, timestamp });
    }

    return warnings;
}

// Attach a rating count (number of u.data rows) to every movie.
// This is an analysis field only; it never feeds the similarity score.
function attachRatingCounts() {
    const counts = new Map();
    for (const rating of ratings) {
        counts.set(rating.itemId, (counts.get(rating.itemId) || 0) + 1);
    }
    for (const movie of movies) {
        movie.ratingCount = counts.get(movie.id) || 0;
    }
}

// Classify each movie as 'head' (top 20% by rating count) or 'long-tail'.
// The 20% boundary is the row index in the deterministic order
// (count desc, title asc, id asc), so the classification is reproducible.
function classifyPopularity() {
    const sorted = [...movies].sort((a, b) => {
        const byCount = b.ratingCount - a.ratingCount;
        if (byCount !== 0) return byCount;
        const byTitle = a.title.localeCompare(b.title);
        if (byTitle !== 0) return byTitle;
        return a.id - b.id;
    });

    const headCount = Math.ceil(movies.length * 0.20);
    catalogStats.headCount = headCount;
    catalogStats.longTailCount = movies.length - headCount;
    catalogStats.headCutoffCount = sorted[headCount - 1].ratingCount;
    catalogStats.distribution = sorted.map(movie => movie.ratingCount);

    sorted.forEach((movie, index) => {
        movie.popularityClass = index < headCount ? 'head' : 'long-tail';
    });
}