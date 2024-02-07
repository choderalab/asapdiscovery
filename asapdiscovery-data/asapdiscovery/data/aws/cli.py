from pathlib import Path

import click

from ..cli import cli

# use these extra kwargs with any option from a settings parameter.
SETTINGS_OPTION_KWARGS = {
    "show_envvar": True,
}


def aws_params(func):
    access_key_id = click.option(
        "--access-key-id",
        type=str,
        default=None,
        envvar="AWS_ACCESS_KEY_ID",
        **SETTINGS_OPTION_KWARGS,
    )
    secret_access_key = click.option(
        "--secret-access-key",
        type=str,
        default=None,
        envvar="AWS_SECRET_ACCESS_KEY",
        **SETTINGS_OPTION_KWARGS,
    )
    session_token = click.option(
        "--session-token",
        type=str,
        default=None,
        envvar="AWS_SESSION_TOKEN",
        **SETTINGS_OPTION_KWARGS,
    )
    region = click.option(
        "--region",
        type=str,
        envvar="AWS_DEFAULT_REGION",
        **SETTINGS_OPTION_KWARGS,
    )
    profile = click.option(
        "--profile",
        type=str,
        envvar="AWS_PROFILE",
        **SETTINGS_OPTION_KWARGS,
    )
    return access_key_id(secret_access_key(session_token(region(profile(func)))))


def s3_params(func):
    bucket = click.option(
        "--bucket", type=str, envvar="AWS_S3_BUCKET", **SETTINGS_OPTION_KWARGS
    )
    prefix = click.option(
        "--prefix", type=str, envvar="AWS_S3_PREFIX", **SETTINGS_OPTION_KWARGS
    )
    endpoint_url = click.option("--endpoint-url", type=str)
    return bucket(prefix(endpoint_url(func)))


@cli.group(help="Commands for interacting with the AWS API")
@aws_params
@click.pass_context
def aws(ctx, access_key_id, secret_access_key, session_token, region, profile):
    from boto3.session import Session

    ctx.ensure_object(dict)

    ctx.obj["access_key_id"] = access_key_id
    ctx.obj["secret_access_key"] = secret_access_key
    ctx.obj["session_token"] = session_token
    ctx.obj["region"] = region
    ctx.obj["profile"] = profile

    # create a boto3 Session and parameterize with keys
    session = Session(
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        aws_session_token=session_token,
        region_name=region,
        profile_name=profile,
    )

    ctx.obj["session"] = session


@aws.group(help="Commands for interacting with AWS S3")
@s3_params
@click.pass_context
def s3(ctx, bucket, prefix, endpoint_url):
    from .s3 import S3

    ctx.obj["s3"] = S3(ctx.obj["session"], bucket, prefix, endpoint_url)


@s3.command(help="Push artifacts to S3")
@click.pass_context
@click.argument("files", type=click.Path(exists=True), nargs=-1)
def push(ctx, files):
    s3 = ctx.obj["s3"]

    click.echo(f"Pushing {len(files)} to 's3://{s3.bucket}/{s3.prefix}'")

    for file in files:
        path = Path(file)
        s3.push_file(path.name, path)
        click.echo(f" : '{file}'")

    click.echo("Done")


@s3.command(help="Pull artifacts from S3")
@click.pass_context
def pull(ctx):
    ...