#!/usr/bin/env python3
# Multi-agent: Executive → Extractor → Researcher → Validator → Synthesizer
# Input: leadership_assessments.csv → Output: exec_reports.json

import json
import pandas as pd
import numpy as np
import re
from typing import Dict, List, Any
from langchain_ollama import OllamaLLM
import spacy
from dataclasses import dataclass

@dataclass
class AgentState:
    exec_data: Dict
    signals: Dict
    benchmarks: List[Dict]
    draft_assessment: Dict
    final_report: Dict

llm = OllamaLLM(model="llama3.2:3b")
nlp = spacy.load("en_core_web_sm")

# TOOLS (Agentic core)
def extract_leadership_signals(exec_comments: str) -> Dict:
    """Extractor Agent tool."""
    doc = nlp(exec_comments.lower())
    signals = {
        'strategic': len(re.findall(r'strateg|vision|future|plan|roadmap', exec_comments, re.I)),
        'people': len(re.findall(r'team|coach|mentor|empower|develop', exec_comments, re.I)),
        'results': len(re.findall(r'result|deliver|execute|achieve', exec_comments, re.I)),
        'change': len(re.findall(r'change|transform|adapt|innovate', exec_comments, re.I)),
        'collab': len(re.findall(r'collab|teamwork|partner|influence', exec_comments, re.I)),
        'competency': len(re.findall(r'build|develop|skill|capability', exec_comments, re.I)),
        'derailers': len(re.findall(r'ego|micro|blame|ignore|resist', exec_comments, re.I))
    }
    return signals

def retrieve_benchmarks(exec_profile: Dict) -> List[Dict]:
    """Researcher Agent tool - similar profiles."""
    # Simulated RAG (we can expand with approaches like simple cosine similarity or FAISS later)
    benchmarks = [
        {'id': 'benchmark_high_strategic', 'profile': 'Strategic Heavyweight', 'success_rate': 0.88, 'risk': 'delegation'},
        {'id': 'benchmark_people_leader', 'profile': 'Team Builder', 'success_rate': 0.92, 'risk': 'execution'},
        {'id': 'benchmark_balanced', 'profile': 'All-Rounder', 'success_rate': 0.85, 'risk': None},
        {'id': 'benchmark_operator', 'profile': 'Results Focus', 'success_rate': 0.78, 'risk': 'vision'}
    ]
    return benchmarks[:3]

# AGENTS
class ExecutiveAgent:
    def __init__(self):
        self.llm = llm

    def plan(self, state: AgentState) -> str:
        prompt = f"""Executive Assessment Planning

Exec: {state.exec_data['exec_id']}
Raw scores: {state.exec_data['capability_scores']}
360: {state.exec_data['360_comments'][:200]}...

Plan analysis steps:
1. Extract signals from 360
2. Retrieve benchmarks
3. Draft assessment
4. Validate gaps

Next action?"""
        return self.llm.invoke(prompt)

class LeadershipAgents:
    @staticmethod
    def extractor(state: AgentState) -> AgentState:
        state.signals = extract_leadership_signals(state.exec_data['360_comments'])
        print(f"✅ Extractor: {state.signals['strategic']} strategic signals")
        return state

    @staticmethod
    def researcher(state: AgentState) -> AgentState:
        state.benchmarks = retrieve_benchmarks(state.exec_data)
        print(f"✅ Researcher: Found {len(state.benchmarks)} benchmarks")
        return state

    @staticmethod
    def synthesizer(state: AgentState) -> AgentState:
        """Deterministic leadership profile synthesis."""

        # 1. Extract capability scores (dict or columns)
        if 'capability_scores' in state.exec_data and isinstance(state.exec_data['capability_scores'], dict):
            scores = state.exec_data['capability_scores']
        else:
            caps = [
                'strategic_thinking', 'results_delivery', 'change_leadership',
                'people_leadership', 'collaboration', 'competency_building'
            ]
            scores = {cap: float(state.exec_data.get(cap, 3.5)) for cap in caps}

        # 2. Normalize 1-5 → 0-1 scale
        norm_scores = {k: round(v / 5.0, 2) for k, v in scores.items()}

        # 3. Identify strengths (top quartile) and risks
        strengths = [
            k.replace('_', ' ').title()
            for k, v in scores.items() if v > 4.2
        ]
        risks = ['Delegation issues'] if state.signals.get('derailers', 0) > 0 else []

        # 4. SMART benchmark matching (not just first!)
        if state.benchmarks:
            def benchmark_score(bench, signals):
                score = 0.0
                bench_id = bench['id'].lower()
                if 'strategic' in bench_id:
                    score += signals.get('strategic', 0) * 0.1
                if 'people' in bench_id:
                    score += signals.get('people', 0) * 0.1
                if 'results' in bench_id or 'operator' in bench_id:
                    score += signals.get('results', 0) * 0.1
                if 'change' in bench_id:
                    score += signals.get('change', 0) * 0.1
                return score

            # Pick BEST match by signal alignment
            scored = [(benchmark_score(b, state.signals), b) for b in state.benchmarks]
            best_scored = max(scored, key=lambda x: x[0])
            _, best_benchmark = best_scored
            benchmark_match = best_benchmark['profile']
            all_benchmarks = [b['profile'] for b in state.benchmarks]
        else:
            benchmark_match = 'No benchmarks available'
            all_benchmarks = []

        # 5. Overall leadership fit (mean normalized competencies)
        overall_fit = round(np.mean(list(norm_scores.values())), 2)

        # 6. Assemble complete deterministic profile
        state.draft_assessment = {
        'exec_id': state.exec_data['exec_id'],
        'raw_scores': scores,
        'normalized_scores': norm_scores,
        'strengths': strengths,
        'risks': risks,
        'benchmark_match': benchmark_match,
        'all_benchmarks': all_benchmarks,
        'overall_fit': overall_fit,
        'signal_summary': state.signals,
    }

        print(f"  ✅ Synthesizer: {overall_fit:.2f} → '{benchmark_match}'")
        return state


def run_agentic_assessment(exec_row: Dict) -> Dict:
    state = AgentState(
        exec_data=exec_row,
        signals={}, benchmarks=[],
        draft_assessment={}, final_report={}
    )

    # Executive orchestration
    exec_agent = ExecutiveAgent()
    plan = exec_agent.plan(state).lower()

    print(f"📋 Executive Plan: {plan[:200]}...")

    # Dynamic execution based on plan
    if 'extract' in plan:
        state = LeadershipAgents.extractor(state)
    if 'benchmark' in plan or 'research' in plan:
        state = LeadershipAgents.researcher(state)
    if 'synthes' in plan or 'assess' in plan:
        state = LeadershipAgents.synthesizer(state)
    prompt = f"""CRITIQUE this leadership assessment using structured reasoning:

RAW DATA:
- Signals: {state.signals}
- Normalized Scores: {state.draft_assessment.get('normalized_scores', {})}
- Strengths: {state.draft_assessment.get('strengths', [])}
- Risks: {state.draft_assessment.get('risks', [])}
- Benchmark: {state.draft_assessment.get('benchmark_match', 'None')}
- Overall Fit: {state.draft_assessment.get('overall_fit', 0)}

VALIDATION CHECKLIST:
1. Do claimed strengths align with high signal counts? (strategic signals 7+ → Strategic strength)
2. Are risks justified by derailers? (derailers > 0 → valid risk)
3. Does benchmark match signal profile? (Strategic Heavyweight → high strategic signals)
4. Is overall_fit realistic? (0.8+ = exceptional, 0.6-0.8 = strong, <0.6 = develop)

Return CLEAN JSON ONLY (no ```, no markdown):

{{
  \"signal_consistency\": true/false,
  \"risk_alignment\": true/false,
  \"benchmark_valid\": true/false,
  \"fit_reasonable\": true/false,
  \"confidence\": 0.0-1.0,
  \"recommendation\": \"hire|develop|review|pass\",
  \"issues\": [\"brief list or empty\"],
  \"narrative\": \"1-2 sentences with reasoning\"
}}"""

    response = llm.invoke(prompt)
    try:
        state.final_report = json.loads(response)
    except:
        state.final_report = {"confidence": 0.5, "recommendation": "review"}

    return {**state.draft_assessment, **state.final_report}  # Merge results!



if __name__ == "__main__":
    df = pd.read_csv('/leadership_assessments.csv')

    reports = []
    for _, row in df.iterrows():
        assessment = run_agentic_assessment(row.to_dict())
        reports.append(assessment)

    # Save
    results_df = pd.DataFrame(reports)
    results_df.to_csv('/leadership_assessments_agentic.csv', index=False)

    with open('/exec_reports.json', 'w') as f:
        json.dump(reports, f, indent=2)

    print("\n🎯 Agentic System COMPLETE!")
    print(results_df[['exec_id', 'overall_fit', 'benchmark_match']].round(2).head())
    print(f"\n📈 Top performer: {results_df.loc[results_df['overall_fit'].idxmax(), 'exec_id']} ({results_df['overall_fit'].max():.2f})")
