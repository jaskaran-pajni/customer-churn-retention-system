"""Static deployment checks; safe to run before a model has been trained."""
from pathlib import Path
import ast

ROOT = Path(__file__).resolve().parents[1]


def test_required_deployment_sources_exist():
    assert (ROOT / 'streamlit_app.py').is_file()
    assert (ROOT / 'requirements.txt').is_file()
    assert (ROOT / '.streamlit/config.toml').is_file()
    assert (ROOT / 'scripts/prepare_deployment.py').is_file()
    assert (ROOT / 'DEPLOY_DAY3.md').is_file()


def test_streamlit_and_preflight_are_parseable():
    for source in ('streamlit_app.py', 'scripts/prepare_deployment.py'):
        ast.parse((ROOT / source).read_text(encoding='utf-8'), filename=source)


def test_gitignore_blocks_private_inputs():
    content = (ROOT / '.gitignore').read_text(encoding='utf-8')
    for pattern in ('data/raw/*', 'data/processed/*', '.streamlit/secrets.toml', 'outputs/*'):
        assert pattern in content
