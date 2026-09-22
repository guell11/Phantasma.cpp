import hljs from 'highlight.js';
import {
	NEWLINE,
	DEFAULT_LANGUAGE,
	LANG_PATTERN,
	AMPERSAND_REGEX,
	LT_REGEX,
	GT_REGEX,
	FENCE_PATTERN,
	TRIM_LEADING_PADDING_REGEX,
	TRIM_TRAILING_PADDING_REGEX
} from '$lib/constants';

export interface IncompleteCodeBlock {
	language: string;
	code: string;
	openingIndex: number;
}

const CODE_BLOCK_EXTENSION_BY_LANGUAGE: Record<string, string> = {
	asciidoc: '.adoc',
	bash: '.sh',
	c: '.c',
	'c++': '.cpp',
	cpp: '.cpp',
	'c#': '.cs',
	csharp: '.cs',
	cs: '.cs',
	css: '.css',
	csv: '.csv',
	cuda: '.cu',
	dart: '.dart',
	go: '.go',
	haskell: '.hs',
	hs: '.hs',
	html: '.html',
	java: '.java',
	javascript: '.js',
	js: '.js',
	json: '.json',
	jsx: '.jsx',
	kotlin: '.kt',
	kt: '.kt',
	latex: '.tex',
	less: '.less',
	lua: '.lua',
	markdown: '.md',
	md: '.md',
	mermaid: '.mmd',
	php: '.php',
	plaintext: '.txt',
	powershell: '.ps1',
	properties: '.properties',
	py: '.py',
	python: '.py',
	r: '.r',
	rb: '.rb',
	ruby: '.rb',
	rust: '.rs',
	rs: '.rs',
	scala: '.scala',
	scss: '.scss',
	sh: '.sh',
	shell: '.sh',
	sql: '.sql',
	svelte: '.svelte',
	svg: '.svg',
	swift: '.swift',
	text: '.txt',
	ts: '.ts',
	tsx: '.tsx',
	txt: '.txt',
	typescript: '.ts',
	vue: '.vue',
	xml: '.xml',
	yaml: '.yaml',
	yml: '.yml',
	zsh: '.sh'
};

/**
 * Returns a predictable download name for a fenced Markdown code block.
 * Unknown language labels deliberately fall back to .txt rather than becoming
 * part of the filename.
 */
export function getCodeBlockFilename(language: string): string {
	const normalizedLanguage = language
		.trim()
		.toLowerCase()
		.replace(/^language-/, '');
	const extension = CODE_BLOCK_EXTENSION_BY_LANGUAGE[normalizedLanguage] ?? '.txt';
	return `code-block${extension}`;
}

/**
 * Downloads code without sending it to the server or evaluating its contents.
 */
export function downloadCodeBlock(code: string, language: string): void {
	if (
		typeof document === 'undefined' ||
		typeof URL === 'undefined' ||
		typeof Blob === 'undefined'
	) {
		return;
	}

	const blob = new Blob([code], { type: 'text/plain;charset=utf-8' });
	const objectUrl = URL.createObjectURL(blob);
	const anchor = document.createElement('a');
	anchor.download = getCodeBlockFilename(language);
	anchor.href = objectUrl;
	anchor.style.display = 'none';
	document.body.appendChild(anchor);

	try {
		anchor.click();
	} finally {
		anchor.remove();
		URL.revokeObjectURL(objectUrl);
	}
}

/**
 * Strips empty lines (whitespace-only) from the start and end of code.
 *
 * Tool call payloads frequently arrive with surrounding whitespace from LLM
 * formatting (`"\nfunction ...\n"`). Preserving those newlines makes hljs emit
 * a leading/trailing empty line that `<pre>` then renders as a phantom row,
 * pushing real content away from the box edge. The trim keeps the body intact
 * so internal blank lines are still rendered as such.
 */
function trimCodePadding(code: string): string {
	return code.replace(TRIM_LEADING_PADDING_REGEX, '').replace(TRIM_TRAILING_PADDING_REGEX, '');
}

function escapeCode(code: string): string {
	return code.replace(AMPERSAND_REGEX, '&amp;').replace(LT_REGEX, '&lt;').replace(GT_REGEX, '&gt;');
}

/**
 * Highlights code using highlight.js
 * @param code - The code to highlight
 * @param language - The programming language
 * @param autoDetect - Fall back to `highlightAuto` when `language` is unknown.
 *   Callers rendering a still-streaming block should pass false: auto-detection
 *   costs ~38ms per call and re-guesses on every chunk, so the language (and
 *   therefore the whole highlight) flickers as the block grows.
 * @returns HTML string with syntax highlighting
 */
export function highlightCode(code: string, language: string, autoDetect = true): string {
	if (!code) return '';

	const trimmed = trimCodePadding(code);

	try {
		const lang = language.toLowerCase();
		const isSupported = hljs.getLanguage(lang);

		if (isSupported) {
			return hljs.highlight(trimmed, { language: lang }).value;
		} else if (autoDetect) {
			return hljs.highlightAuto(trimmed).value;
		} else {
			return escapeCode(trimmed);
		}
	} catch {
		// Fallback to escaped plain text
		return escapeCode(trimmed);
	}
}

export { trimCodePadding };

/**
 * Detects if markdown ends with an incomplete code block (opened but not closed).
 * Returns the code block info if found, null otherwise.
 * @param markdown - The raw markdown string to check
 * @returns IncompleteCodeBlock info or null
 */
export function detectIncompleteCodeBlock(markdown: string): IncompleteCodeBlock | null {
	// Count all code fences in the markdown
	// A code block is incomplete if there's an odd number of ``` fences
	const fencePattern = new RegExp(FENCE_PATTERN.source, FENCE_PATTERN.flags);
	const fences: number[] = [];
	let fenceMatch;

	while ((fenceMatch = fencePattern.exec(markdown)) !== null) {
		// Store the position after the ```
		const pos = fenceMatch[0].startsWith(NEWLINE) ? fenceMatch.index + 1 : fenceMatch.index;
		fences.push(pos);
	}

	// If even number of fences (including 0), all code blocks are closed
	if (fences.length % 2 === 0) {
		return null;
	}

	// Odd number means last code block is incomplete
	// The last fence is the opening of the incomplete block
	const openingIndex = fences[fences.length - 1];
	const afterOpening = markdown.slice(openingIndex + 3);

	// Extract language and code content
	const langMatch = afterOpening.match(LANG_PATTERN);
	const language = langMatch?.[1] || DEFAULT_LANGUAGE;
	const codeStartIndex = openingIndex + 3 + (langMatch?.[0]?.length ?? 0);
	const code = markdown.slice(codeStartIndex);

	return {
		language,
		code,
		openingIndex
	};
}
