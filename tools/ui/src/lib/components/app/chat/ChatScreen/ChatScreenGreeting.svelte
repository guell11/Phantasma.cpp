<script lang="ts">
	import { serverStore } from '$lib/stores/server.svelte';

	interface Props {
		isEmpty: boolean;
	}

	let { isEmpty = false }: Props = $props();
</script>

<div
	class={[
		'pointer-events-none mb-4 hidden px-4 text-center text-balance',
		isEmpty && 'mb-[calc(50dvh-8rem)] md:mb-6 pointer-events-auto block!'
	]}
>
		<div class="mx-auto mb-4 flex w-fit items-center gap-2 rounded-full border border-violet-400/20 bg-violet-400/8 px-3 py-1 text-xs font-medium tracking-[0.16em] text-violet-300 uppercase shadow-[0_0_32px_rgba(139,92,246,.12)]">
			<span class="h-1.5 w-1.5 rounded-full bg-cyan-300 shadow-[0_0_12px_rgba(34,211,238,.9)]"></span>
			local inference online
		</div>

		<h1 class="mb-2 text-3xl font-semibold tracking-[-0.04em] md:text-5xl">
			<span class="phantasma-gradient-text">phantasma.cpp</span>
		</h1>

	<p class="text-muted-foreground md:text-lg">
			{serverStore.props?.modalities?.audio ? 'Record audio, type a message ' : 'Type a message'} or drop a file.
			<span class="hidden md:inline"> Fast local inference, zero cloud required.</span>
		</p>
</div>
