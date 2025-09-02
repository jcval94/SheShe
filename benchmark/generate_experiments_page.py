import pandas as pd
from pathlib import Path

BENCHMARK_DIR = Path(__file__).parent
DOC_FILE = BENCHMARK_DIR.parent / 'docs' / 'experiments_benchmarks.html'

DESCRIPTIONS = {
    'cheche_vs_shushu_ab_test.csv': (
        'Runtime comparison between CheChe.fit and a ShuShu-style frontier computation '
        '(see <code>cheche_vs_shushu_ab_test.py</code> for details).'
    ),
    'newton_vs_gradient_ab_test.csv': (
        'Gradient ascent versus Newton trust-region on a quadratic objective; includes evaluation counts '
        '(script: <code>newton_vs_gradient_ab_test.py</code>).'
    ),
    'percentile_drop_ab_test.csv': (
        'Vectorised <code>find_percentile_drop</code> compared to the previous loop-based version '
        '(script: <code>percentile_drop_ab_test.py</code>).'
    ),
    'parallel_jobs_ab_test.csv': (
        'ModalScoutEnsemble fitted sequentially (<code>n_jobs=1</code>) versus in parallel (<code>n_jobs=-1</code>) '
        '(script: <code>parallel_jobs_ab_test.py</code>).' 
    ),
    'ray_mode_ab_test.csv': (
        "ModalBoundaryClustering ray modes 'grid' and 'grad' compared for speed and accuracy "
        '(script: <code>ray_mode_ab_test.py</code>).'
    ),
    'numba_finite_diff_ab_test.csv': (
        'Effect of numba acceleration on finite-difference gradients '
        '(results generated externally).'
    ),
    'large_df_benchmark_results.csv': (
        'Benchmark measuring fit time scaling with large datasets '
        '(script: <code>large_df_benchmark.py</code>).'
    ),
    'stop_criteria_results.csv': (
        'Impact of different stop criteria on convergence behaviour '
        '(script: <code>profile_fit.py</code>).'
    ),
    'subspace_ab_results.csv': (
        'Comparison of subspace search strategies.'
    ),
    'unsupervised_results_summary.csv': (
        'Summary of unsupervised modal clustering benchmarks.'
    ),
    'fit_profile.csv': (
        'Profiling data for <code>fit</code> performance (script: <code>profile_fit.py</code>).'
    ),
}

def main():
    parts = [
        '---',
        'layout: default',
        'title: "Experiments & Benchmarks"',
        'nav_order: 98',
        '---',
        '',
        '<link rel="stylesheet" href="style.css">',
        '<h1>Experiments & Benchmarks</h1>',
        '<p>This page summarises results from scripts in the <code>benchmark/</code> directory. '
        'Each script runs an A/B style comparison with several repetitions and reports the mean '
        'runtime, a speedup ratio and any additional metrics specific to the experiment.</p>'
    ]

    for csv_path in sorted(BENCHMARK_DIR.glob('*.csv')):
        name = csv_path.name
        df = pd.read_csv(csv_path)
        desc = DESCRIPTIONS.get(name, '')
        parts.append(f'<h2>{name}</h2>')
        if desc:
            parts.append(f'<p>{desc}</p>')
        parts.append('<div class="scroll-table">')
        parts.append(df.to_html(index=False))
        parts.append('</div>')

    DOC_FILE.write_text('\n'.join(parts), encoding='utf-8')

if __name__ == '__main__':
    main()
