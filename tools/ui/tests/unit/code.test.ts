import { describe, expect, it } from 'vitest';
import { getCodeBlockFilename, highlightCode, trimCodePadding } from '$lib/utils/code';

describe('getCodeBlockFilename', () => {
	it('maps common fenced languages to conventional extensions', () => {
		expect(getCodeBlockFilename('html')).toBe('code-block.html');
		expect(getCodeBlockFilename('javascript')).toBe('code-block.js');
		expect(getCodeBlockFilename('typescript')).toBe('code-block.ts');
		expect(getCodeBlockFilename('python')).toBe('code-block.py');
		expect(getCodeBlockFilename('mermaid')).toBe('code-block.mmd');
		expect(getCodeBlockFilename('svg')).toBe('code-block.svg');
	});

	it('normalizes aliases and casing', () => {
		expect(getCodeBlockFilename(' C++ ')).toBe('code-block.cpp');
		expect(getCodeBlockFilename('language-PowerShell')).toBe('code-block.ps1');
		expect(getCodeBlockFilename('yml')).toBe('code-block.yml');
	});

	it('falls back to text without embedding an unknown language in the filename', () => {
		expect(getCodeBlockFilename('../../unexpected')).toBe('code-block.txt');
		expect(getCodeBlockFilename('')).toBe('code-block.txt');
	});
});

describe('trimCodePadding', () => {
	it('removes a single leading newline', () => {
		expect(trimCodePadding('\nfunction foo() {}')).toBe('function foo() {}');
	});

	it('removes multiple leading newlines', () => {
		expect(trimCodePadding('\n\n\nfunction foo() {}')).toBe('function foo() {}');
	});

	it('removes whitespace-only leading lines', () => {
		expect(trimCodePadding('\n  \n\t\nfunction foo() {}')).toBe('function foo() {}');
	});

	it('removes a single trailing newline', () => {
		expect(trimCodePadding('function foo() {}\n')).toBe('function foo() {}');
	});

	it('removes multiple trailing newlines', () => {
		expect(trimCodePadding('function foo() {}\n\n\n')).toBe('function foo() {}');
	});

	it('removes whitespace-only trailing lines', () => {
		expect(trimCodePadding('function foo() {}\n  \n\t\n')).toBe('function foo() {}');
	});

	it('removes newlines on both sides at once', () => {
		expect(trimCodePadding('\nfunction foo() {}\n')).toBe('function foo() {}');
	});

	it('preserves internal blank lines', () => {
		expect(trimCodePadding('\nfunction foo() {\n\n  return 1;\n}\n')).toBe(
			'function foo() {\n\n  return 1;\n}'
		);
	});

	it('drops a leading whitespace-only line but keeps following code intact', () => {
		expect(trimCodePadding('  \nfunction foo() {}')).toBe('function foo() {}');
	});

	it('passes through already-trimmed input unchanged', () => {
		expect(trimCodePadding('function foo() {}')).toBe('function foo() {}');
		expect(trimCodePadding('function foo() {\n  return 1;\n}')).toBe(
			'function foo() {\n  return 1;\n}'
		);
	});

	it('returns empty string when input is whitespace only', () => {
		expect(trimCodePadding('\n\n\n')).toBe('');
		expect(trimCodePadding('\n  \n\t\n')).toBe('');
	});
});

describe('highlightCode', () => {
	it('returns empty string for empty input', () => {
		expect(highlightCode('', 'javascript')).toBe('');
	});

	it('does not produce a leading newline in the highlighted html', () => {
		const html = highlightCode('\nfunction multiply(a, b) {\n  return a * b;\n}\n', 'javascript');
		expect(html.startsWith('\n')).toBe(false);
		expect(html.startsWith(' ')).toBe(false);
	});

	it('does not produce a trailing newline in the highlighted html', () => {
		const html = highlightCode('\nfunction foo() {}\n', 'javascript');
		expect(html.endsWith('\n')).toBe(false);
	});

	it('preserves internal blank lines in highlighted code', () => {
		const html = highlightCode('\nfunction foo() {\n\n  return 1;\n}\n', 'javascript');
		expect(html).toContain('\n\n');
	});

	it('produces the same body for framed and unframed input', () => {
		const trimmed = highlightCode('function foo() {}', 'javascript');
		const framed = highlightCode('\nfunction foo() {}\n', 'javascript');
		expect(framed).toBe(trimmed);
	});

	it('auto-detects an unknown language by default', () => {
		const html = highlightCode('const answer = 42;', 'not-a-language');
		expect(html).toContain('hljs-');
	});

	it('escapes instead of auto-detecting when autoDetect is false', () => {
		const html = highlightCode('const answer = 42;', 'not-a-language', false);
		expect(html).not.toContain('hljs-');
		expect(html).toBe('const answer = 42;');
	});

	it('still highlights a known language when autoDetect is false', () => {
		const html = highlightCode('const answer = 42;', 'javascript', false);
		expect(html).toContain('hljs-');
	});

	it('escapes html metacharacters when falling back to plain text', () => {
		const html = highlightCode('<script>a && b</script>', 'not-a-language', false);
		expect(html).toBe('&lt;script&gt;a &amp;&amp; b&lt;/script&gt;');
	});
});
