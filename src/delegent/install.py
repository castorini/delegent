"""Optional: Delegent inside the agent you already use, instead of the `delegent claude` launcher.

`delegent install claude` and `delegent install codex` set up your own Claude Code or Codex so that it follows the same
rule as the standalone harness: the main loop keeps your frontier model and your login, and every subagent runs on the
SLM. Both use the attached relay (`delegent serve`, see relay.py), which serves only the calls that name the SLM and
passes everything else through untouched.

- Claude Code: three variables in the `env` of your settings.json point Claude Code at the relay and give the SLM the
  subagent and small-model roles (the same roles `delegent claude` assigns), the coding instruction goes into your
  user-level CLAUDE.md, and a SessionStart hook starts the relay when it is not running.
- Codex: a profile (`codex -p delegent`) that sends model calls through the relay, makes the SLM the default subagent
  model, and carries the instruction, plus a copy of Codex's model catalog with the SLM added and the subagent tools
  that pass tasks as plain text. Your config.toml is not edited.

With a corpus (`--corpus`), either agent also gets the search tools of the harness over MCP, served by the same
relay: `search`, `get_document` and `Delegate`, whose SLM subagent searches in its own loop, and the search part of
the instruction. This is what `delegent claude --corpus` (merged mode) gives the launcher's session.

`delegent uninstall claude|codex` removes exactly what was added.
"""
import copy
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from .agent import instruction
from .mcp import SERVER_NAME

BEGIN, END = '<!-- delegent:begin -->', '<!-- delegent:end -->'
CLAUDE_ENV = ('ANTHROPIC_BASE_URL', 'CLAUDE_CODE_SUBAGENT_MODEL', 'ANTHROPIC_DEFAULT_HAIKU_MODEL')
HOOK_MARK = 'delegent.cli serve --ensure'
PROFILE = 'delegent'
DEFAULT_ANTHROPIC = 'https://api.anthropic.com'


class InstallError(RuntimeError):
    pass


def serve_command(args_list):
    """The command that starts the attached relay unless it is already up (used by the Claude Code hook)."""
    return shlex.join([sys.executable, '-m', 'delegent.cli', 'serve', '--ensure', *args_list])


def tools_command(args_list):
    """The MCP server an agent starts for the search tools: a stdio bridge to the relay's /mcp, which also starts the
    relay when it is not running (an agent connects its MCP servers before any hook or model call)."""
    return [sys.executable, '-m', 'delegent.cli', 'mcp', *args_list]


# ---------------------------------------------------------------------------------------------------------------------
# Claude Code
# ---------------------------------------------------------------------------------------------------------------------
def claude_dir(config_dir=None):
    return Path(config_dir or os.environ.get('CLAUDE_CONFIG_DIR') or Path.home() / '.claude').expanduser()


def _load_settings(path):
    if not path.exists():
        return {}
    try:
        settings = json.loads(path.read_text() or '{}')
    except ValueError as exc:
        raise InstallError(f'{path} is not valid JSON ({exc}); fix it first') from exc
    if not isinstance(settings, dict):
        raise InstallError(f'{path} does not hold a JSON object')
    return settings


def _without_block(text):
    """The text without the Delegent block."""
    if BEGIN not in text:
        return text
    head, _, rest = text.partition(BEGIN)
    _, _, tail = rest.partition(END)
    return (head.rstrip('\n') + '\n' + tail.lstrip('\n')).strip('\n') + ('\n' if (head + tail).strip() else '')


def _is_ours(entry):
    return any(HOOK_MARK in str(h.get('command', '')) for h in entry.get('hooks') or [] if isinstance(h, dict))


def claude_mcp(action, config_dir=None, claude_bin=None, command=None):
    """Register (or remove) the relay's search tools as a user-scope MCP server, through Claude Code's own command."""
    claude = claude_bin or shutil.which('claude')
    if not claude:
        raise InstallError('Claude Code (claude) not found; install it or pass --claude-bin')
    env = dict(os.environ)
    if config_dir:
        env['CLAUDE_CONFIG_DIR'] = str(claude_dir(config_dir))
    run = lambda *args: subprocess.run([claude, 'mcp', *args], capture_output=True, text=True, env=env)  # noqa: E731
    known = run('get', SERVER_NAME)
    ours = known.returncode == 0 and 'delegent.cli' in known.stdout
    if known.returncode == 0 and not ours:   # the user's own server of that name
        if action == 'add':
            raise InstallError(f'Claude Code already has an MCP server named "{SERVER_NAME}"; remove or rename it')
        return False
    if ours:
        run('remove', '--scope', 'user', SERVER_NAME)
    if action == 'add':
        got = run('add', '--scope', 'user', SERVER_NAME, '--', *command)
        if got.returncode != 0:
            raise InstallError(f'claude mcp add failed: {(got.stderr or got.stdout)[-300:]}')
        return True
    return ours


def install_claude(relay_url, slm_model, hook_command=None, config_dir=None, force=False, search=None,
                   claude_bin=None):
    """Point an existing Claude Code at the attached relay -> list of what was changed."""
    home = claude_dir(config_dir)
    home.mkdir(parents=True, exist_ok=True)
    path = home / 'settings.json'
    settings = _load_settings(path)
    env = settings.setdefault('env', {})
    current = env.get('ANTHROPIC_BASE_URL')
    if current and current.rstrip('/') not in (relay_url, DEFAULT_ANTHROPIC) and not force:
        raise InstallError(f'{path} already sets ANTHROPIC_BASE_URL={current}; Delegent would replace that gateway '
                           '(pass --force to do so)')
    if path.exists() and not (home / 'settings.json.delegent-backup').exists():
        shutil.copy2(path, home / 'settings.json.delegent-backup')
    # the same roles as `delegent claude`: subagents and the small/fast model run on the SLM
    env.update(ANTHROPIC_BASE_URL=relay_url, CLAUDE_CODE_SUBAGENT_MODEL=slm_model,
               ANTHROPIC_DEFAULT_HAIKU_MODEL=slm_model)
    done = [f'{path}: env.' + ', env.'.join(CLAUDE_ENV)]
    shell = os.environ.get('ANTHROPIC_BASE_URL')
    if shell and shell.rstrip('/') not in (relay_url, DEFAULT_ANTHROPIC):
        done.append(f'note: your shell exports ANTHROPIC_BASE_URL={shell}, which may take precedence over settings.json')
    hooks = settings.setdefault('hooks', {})
    start = [e for e in hooks.get('SessionStart') or [] if not _is_ours(e)]
    if hook_command:
        start.append({'hooks': [{'type': 'command', 'command': hook_command}]})
        done.append(f'{path}: a SessionStart hook that starts the relay when it is not running')
    if start:
        hooks['SessionStart'] = start
    else:
        hooks.pop('SessionStart', None)
    if not hooks:
        settings.pop('hooks')
    path.write_text(json.dumps(settings, indent=2) + '\n')
    memory = home / 'CLAUDE.md'
    kept = _without_block(memory.read_text() if memory.exists() else '')
    tools = {'Agent', 'Delegate'} if search else {'Agent'}   # as `delegent claude` with and without --corpus
    block = f'{BEGIN}\n{instruction(tools).strip()}\n{END}\n'
    memory.write_text((kept.rstrip('\n') + '\n\n' if kept.strip() else '') + block)
    done.append(f'{memory}: the Delegent instruction (between the delegent markers)')
    if search:
        claude_mcp('add', config_dir, claude_bin, search)
        done.append(f'MCP server "{SERVER_NAME}" (user scope): search, get_document and Delegate, served by the relay')
    else:   # a reinstall without a corpus leaves no stale server behind
        try:
            claude_mcp('remove', config_dir, claude_bin)
        except InstallError:
            pass
    return done


def uninstall_claude(config_dir=None, claude_bin=None):
    home = claude_dir(config_dir)
    path, done = home / 'settings.json', []
    try:
        if claude_mcp('remove', config_dir, claude_bin):
            done.append(f'MCP server "{SERVER_NAME}": removed')
    except InstallError:
        pass
    if path.exists():
        settings = _load_settings(path)
        env = settings.get('env') or {}
        removed = [k for k in CLAUDE_ENV if env.pop(k, None) is not None]
        if 'env' in settings and not env:
            settings.pop('env')
        hooks = settings.get('hooks') or {}
        if 'SessionStart' in hooks:
            hooks['SessionStart'] = [e for e in hooks['SessionStart'] if not _is_ours(e)]
            if not hooks['SessionStart']:
                hooks.pop('SessionStart')
        if 'hooks' in settings and not hooks:
            settings.pop('hooks')
        path.write_text(json.dumps(settings, indent=2) + '\n')
        if removed:
            done.append(f'{path}: removed env.' + ', env.'.join(removed) + ' and the relay hook')
    memory = home / 'CLAUDE.md'
    if memory.exists() and BEGIN in memory.read_text():
        kept = _without_block(memory.read_text())
        if kept.strip():
            memory.write_text(kept)
        else:
            memory.unlink()
        done.append(f'{memory}: removed the Delegent instruction')
    return done


# ---------------------------------------------------------------------------------------------------------------------
# Codex
# ---------------------------------------------------------------------------------------------------------------------
def codex_dir(codex_home=None):
    return Path(codex_home or os.environ.get('CODEX_HOME') or Path.home() / '.codex').expanduser()


def slm_catalog_entry(models, slm_model, context_tokens):
    """A model-catalog entry for the SLM. Codex only spawns subagents on models in its catalog, and the entry also
    tells it the SLM's context window, so a long subagent compacts instead of overflowing. It starts from the plainest
    bundled entry and turns off what a local model served by vLLM does not have (reasoning levels, service tiers,
    images, hosted web search, the freeform patch tool)."""
    plain = [m for m in models if 'tool_mode' not in m and 'multi_agent_version' not in m] or models
    entry = copy.deepcopy(plain[-1])
    entry.update(slug=slm_model, display_name=slm_model, description='Local small language model (Delegent).',
                 priority=999, visibility='hide', context_window=context_tokens, max_context_window=context_tokens,
                 supported_reasoning_levels=[], default_reasoning_level=None, support_verbosity=False,
                 additional_speed_tiers=[], service_tiers=[], upgrade=None, availability_nux=None,
                 apply_patch_tool_type=None, shell_type='shell_command', web_search_tool_type='text',
                 supports_search_tool=False, supports_image_detail_original=False, input_modalities=['text'],
                 experimental_supported_tools=[])
    for key in ('default_verbosity', 'default_service_tier'):
        entry.pop(key, None)
    return entry


def plain_subagent_tasks(model):
    """Codex's newer subagent tools (multi_agent_version v2) hand the task to a subagent as content that only OpenAI's
    servers can read, and fork the parent's whole history into it by default. A local SLM needs the task as plain text
    and, as in the standalone harness, a fresh context, which is what the v1 tools do."""
    if 'multi_agent_version' in model:
        model['multi_agent_version'] = 'v1'
    return model


def _toml(value):
    return json.dumps(value)   # a JSON string is a valid TOML basic string


def codex_profile(relay_url, slm_model, catalog_path, env_key='OPENAI_API_KEY', openai_auth=False, search=None):
    auth = 'requires_openai_auth = true' if openai_auth else f'env_key = {_toml(env_key)}'
    text = instruction({'Codex', 'Delegate'} if search else {'Codex'}).strip()
    # a delegated search runs a whole subagent, so it needs far longer than Codex's default tool timeout
    tools = f"""
[mcp_servers.{SERVER_NAME}]
command = {_toml(search[0])}
args = {_toml(search[1:])}
tool_timeout_sec = 3600
supports_parallel_tool_calls = true
""" if search else ''
    if "'''" in text:
        raise InstallError("the instruction cannot contain '''")
    return f"""# Written by `delegent install codex`; remove with `delegent uninstall codex`.
# Use with: codex -p {PROFILE}
# The main agent keeps its model and its credentials; subagents run on the SLM behind the Delegent relay.
model_provider = "delegent"
model_catalog_json = {_toml(str(catalog_path))}
developer_instructions = '''
{text}
'''

[model_providers.delegent]
name = "Delegent relay"
base_url = {_toml(relay_url.rstrip('/') + '/codex')}
{auth}
wire_api = "responses"
supports_websockets = false

[agents]
default_subagent_model = {_toml(slm_model)}
{tools}"""


def install_codex(relay_url, slm_model, context_tokens=131072, codex_home=None, codex_bin=None,
                  env_key='OPENAI_API_KEY', openai_auth=False, search=None):
    """Write the `delegent` Codex profile and its model catalog -> list of what was written."""
    home = codex_dir(codex_home)
    codex = codex_bin or shutil.which('codex')
    if not codex:
        raise InstallError('Codex (codex) not found; install it or pass --codex-bin')
    env = {**os.environ, 'CODEX_HOME': str(home)}
    got = subprocess.run([codex, 'debug', 'models'], capture_output=True, text=True, env=env, cwd=str(Path.home()))
    try:
        catalog = json.loads(got.stdout)
        models = [m for m in catalog['models'] if m.get('slug') != slm_model]
    except (ValueError, KeyError, TypeError) as exc:
        raise InstallError(f'could not read Codex\'s model catalog (codex debug models): {got.stderr[-300:]}') from exc
    entry = slm_catalog_entry(models, slm_model, context_tokens)
    for model in models:   # see plain_subagent_tasks
        plain_subagent_tasks(model)
    catalog['models'] = models + [entry]
    home.mkdir(parents=True, exist_ok=True)
    catalog_path, profile_path = home / 'delegent.models.json', home / f'{PROFILE}.config.toml'
    catalog_path.write_text(json.dumps(catalog))
    profile_path.write_text(codex_profile(relay_url, slm_model, catalog_path, env_key, openai_auth, search))
    check = subprocess.run([codex, 'debug', 'models', '-c', f'model_catalog_json={_toml(str(catalog_path))}'],
                           capture_output=True, text=True, env=env, cwd=str(Path.home()))
    if check.returncode != 0 or f'"{slm_model}"' not in check.stdout:
        problem = (check.stderr or check.stdout)[-400:]
        catalog_path.unlink()
        profile_path.unlink()
        raise InstallError(f'this Codex version did not accept the profile or the SLM catalog entry: {problem}')
    return [f'{profile_path}: the profile (provider through the relay, default subagent model, instruction'
            + (', search tools over MCP)' if search else ')'),
            f'{catalog_path}: Codex\'s model catalog plus {slm_model}, with plain-text subagent tasks']


def uninstall_codex(codex_home=None):
    home, done = codex_dir(codex_home), []
    for path in (home / f'{PROFILE}.config.toml', home / 'delegent.models.json'):
        if path.exists():
            path.unlink()
            done.append(f'{path}: removed')
    return done
