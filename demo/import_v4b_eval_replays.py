"""Convert existing V4b evaluation RESULTS to honest, offline Streamlit replay files.

Usage (from shopping-agent project root):
    python demo/import_v4b_eval_replays.py

This script NEVER calls an API or changes app/agents. It creates/replaces only
'demo/fixtures/eval_v4b_*.json'. Each output represents a saved END-TO-END
EVALUATION RECORD, not an individual node trace or a newly run Agent.
"""
from __future__ import annotations
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / 'reports' / 'agent_eval_real_v4b_probe.json'
TARGET = ROOT / 'demo' / 'fixtures'


def clean(value, fallback='—'):
    if value is None or value == '':
        return fallback
    return value


def safe_id(value):
    return re.sub(r'[^a-zA-Z0-9_-]', '_', str(value))[:72]


def main():
    if not SOURCE.is_file():
        raise SystemExit(f'未找到历史报告：{SOURCE}')
    report = json.loads(SOURCE.read_text(encoding='utf-8'))
    rows = report.get('results')
    if not isinstance(rows, list):
        raise SystemExit('报告没有 results 列表，未写入任何文件')

    TARGET.mkdir(parents=True, exist_ok=True)
    written = 0
    for index, r in enumerate(rows, 1):
        if not isinstance(r, dict):
            continue
        if r.get('task_success') is not True:
            print(f'跳过 {r.get("scenario_id", index)}：未确认 task_success=True')
            continue
        if not isinstance(r.get('answer'), str) or not r.get('answer').strip():
            print(f'跳过 {r.get("scenario_id", index)}：无有效回答')
            continue
        sid = str(r.get('scenario_id') or f'record_{index}')
        route = clean(r.get('actual_route'))
        tool = clean(r.get('actual_tool'))
        latency = r.get('latency_ms')
        # A single factual EVALUATION SUMMARY row; no fabricated agent nodes.
        event = {
            'node': 'Agent end-to-end (eval)',
            'event_type': f'eval_summary; actual_route={route}',
            'tool': tool,
            'latency_ms': latency if isinstance(latency, (int, float)) else None,
            'status': 'task_success=True',
        }
        record = {
            'source': 'recorded_eval_summary',
            'source_file': str(SOURCE.relative_to(ROOT)),
            'scenario': sid,
            'scenario_type': r.get('scenario_type'),
            'query': clean(r.get('message')),
            'answer': r['answer'],
            'actual_route': route,
            'actual_tool': tool,
            'task_success': True,
            'latency_ms': event['latency_ms'],
            'events': [event],
            'limitations': '只包含评测记录中的实际路由、工具、回答和端到端延迟；没有逐节点 Agent Trace。',
        }
        target = TARGET / f'eval_v4b_{safe_id(sid)}.json'
        target.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(f'已写入 {target.relative_to(ROOT)} | route={route} | tool={tool} | total_latency_ms={event["latency_ms"]}')
        written += 1
    print(f'完成：写入 {written} 条真实历史评测回放，不调用 Groq API。')
    if written == 0:
        print('没有成功记录；请检查 results 字段，未添加虚构数据。')
    print('公开 Demo 前请审阅 query/answer，确保不含私人信息。')


if __name__ == '__main__':
    main()
