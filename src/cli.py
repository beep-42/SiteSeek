import click
import numpy

from Database import Database, Result
import json
from sys import stdout
from timeit import default_timer as timer


def convert_to_json(results: list[Result]) -> str:
    """Converts list of Results to a dict of dicts (index to object) and subsequently serializes it using json.dumps."""

    obj = {i: r.__dict__ for i,r in enumerate(results)}
    for one in obj:
        single_result = obj[one]
        for field in single_result:
            if isinstance(single_result[field], numpy.ndarray):
                single_result[field] = single_result[field].tolist()

    return json.dumps(obj)

@click.group()
def cli():
    pass

@cli.command(short_help="Build the database from a directory")
@click.argument('directory', type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True))
@click.argument('output', default='database.lzma',
                type=click.Path(exists=False, file_okay=True, dir_okay=False, writable=True), required=False)
@click.option('-t', '--threads', default=-1, help='Number of threads to use (default cpu_count)')
@click.option('--compress', default=False, is_flag=True, help='Compress the database file')
def build(directory, output, threads, compress):
    """
    Builds the database from all files in the given directory.

    :directory: Directory where files are located.
    :output: Path to output the built database.
    """
    click.echo(click.style('Building database...', fg='green', bold=True))
    db = Database()
    try:
        db.add_from_directory(directory, n_jobs=threads)
        click.echo(click.style('Done!', fg='green', bold=True))
        db.save(output, compress=compress)
        click.echo(click.style(f'Saved to {output}', fg='yellow', bold=False))
    except Exception as e:
        click.echo(click.style(f"Failed! Exception: {e}", fg='red', bold=True))

@cli.command()
@click.argument('database', type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True))
@click.argument('query', type=click.STRING)
@click.argument('site', type=click.STRING)
@click.option('-lr', '--likelihood-ratio', default=0.9, type=click.FloatRange(0, 2), help='The required likelihood ratio')
@click.option('-kt', '--kmer-threshold', default=11, type=click.FloatRange(0, 20), help='The minimum required kmer similarity')
@click.option('--no-clustering', is_flag=True, show_default=True, default=False, help='Disables the clustering and relies solely on the RANSAC algorithm.')
@click.option('--ransac-min', default=15, show_default=True, type=click.IntRange(1, 1500), help='The minimum successful expansions to terminate the RANSAC algorithm.')
@click.option('--no-refine', default=False, is_flag=True, type=click.BOOL, help='Skip ICP refinement.')
@click.option('-o', '--output', help='Output file for the results', required=False, type=click.File(mode='w'),
              default=stdout)
@click.option('--compressed', default=False, is_flag=True, help='Uncompress the database file.')
@click.option('--output-json', default=False, is_flag=True, help='Output the results as JSON.')
# @click.option('--format', help='Output format (JSON or CSV)', required=False)
def search(database, query, site, likelihood_ratio, kmer_threshold, no_clustering, ransac_min, no_refine, output, compressed, output_json):
    """
    Searches the given database and returns JSON output.

    :database: Path to the database file.
    :query: PDB ID of the query structure
    :site: Comma-delimited list of residue indices.
    """

    # try:
    site = site.split(',') # [int(x) for x in site.split(',')]
    # except Exception as e:
    #     click.echo(
    #         click.style(f"Invalid site definition! Use a comma separated list of site indices. Example: 1,2,3.",
    #                     fg='red', bold=True),
    #         err=True
    #     )
    #     return

    click.echo(click.style('Loading the database...', fg='yellow', bold=False), err=True)
    db: Database = Database.load(database, compressed=compressed)
    click.echo(click.style('Running the search...', fg='green', bold=True), err=True)
    try:
        t = timer()
        results = db.search(query, site=site, k_mer_similarity_threshold=kmer_threshold, lr=likelihood_ratio,
                            skip_clustering=no_clustering, skip_icp=no_refine, ransac_min=ransac_min)
        click.echo(click.style(f'Done in {timer() - t :.{2}f} seconds. Found {len(results)} results.', fg='cyan', bold=True), err=True)

        # output.write(json.dumps(results))
        if output_json:
            output.write(convert_to_json(results))
        else:
            output.write(Result.get_header())
            output.write(''.join([str(x) for x in sorted(results, key=lambda x: x.score, reverse=True)]))

    except Exception as e:
        click.echo(click.style(f"Failed! Exception: {e}", fg='red', bold=True))

if __name__ == '__main__':
    cli()