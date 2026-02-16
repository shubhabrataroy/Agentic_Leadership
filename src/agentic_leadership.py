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
    # Simulated RAG (expand with FAISS later)
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
        # Handle capability_scores as dict OR individual columns
        if 'capability_scores' in state.exec_data and isinstance(state.exec_data['capability_scores'], dict):
            scores = state.exec_data['capability_scores']
        else:
            # Fallback to individual columns
            caps = ['strategic_thinking', 'results_delivery', 'change_leadership',
                    'people_leadership', 'collaboration', 'competency_building']
            scores = {cap: state.exec_data.get(cap, 3.5) for cap in caps}

        # Normalize (1-5 → 0-1)
        norm_scores = {k: round(float(v)/5, 2) for k, v in scores.items()}

        strengths = [k.replace('_', ' ').title() for k, v in scores.items()
                    if float(v) > 4.2]
        risks = ['Delegation issues'] if state.signals.get('derailers', 0) > 0 else []

        state.draft_assessment = {
            'exec_id': state.exec_data['exec_id'],
            'raw_scores': scores,
            'normalized': norm_scores,
            'strengths': strengths,
            'risks': risks,
            'benchmark_match': state.benchmarks[0]['profile'] if state.benchmarks else None,
            'overall_fit': round(np.mean([norm_scores[k] for k in norm_scores]), 2),
            'signal_summary': state.signals
        }
        print(f"  ✅ Synthesizer: Overall {state.draft_assessment['overall_fit']:.2f}")
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

    # Always validate
    prompt = f"""Validate assessment for {exec_row['exec_id']}:\n{json.dumps(state.draft_assessment, indent=2)}\n\nFinal report?"""
    state.final_report = llm.invoke(prompt)

    return state.draft_assessment


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
