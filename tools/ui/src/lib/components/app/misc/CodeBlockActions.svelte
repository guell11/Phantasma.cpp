<script lang="ts">
	import { Download, Eye } from '@lucide/svelte';
	import { ActionIcon, ActionIconCopyToClipboard } from '$lib/components/app';
	import { FileTypeText } from '$lib/enums';
	import { downloadCodeBlock } from '$lib/utils';

	interface Props {
		code: string;
		language: string;
		disabled?: boolean;
		onPreview?: (code: string, language: string) => void;
	}

	let { code, language, disabled = false, onPreview }: Props = $props();

	const showPreview = $derived(language?.toLowerCase() === FileTypeText.HTML);
</script>

<div class="code-block-actions">
	<ActionIconCopyToClipboard
		text={code}
		canCopy={!disabled}
		ariaLabel={disabled ? 'Code incomplete' : 'Copy code'}
	/>

	<ActionIcon
		{disabled}
		icon={Download}
		onclick={() => {
			if (!disabled) downloadCodeBlock(code, language);
		}}
		tooltip={disabled ? 'Code incomplete' : 'Download code'}
	/>

	{#if showPreview}
		<ActionIcon
			{disabled}
			icon={Eye}
			onclick={() => onPreview!(code, language)}
			tooltip={disabled ? 'Code incomplete' : 'Preview code'}
		/>
	{/if}
</div>
