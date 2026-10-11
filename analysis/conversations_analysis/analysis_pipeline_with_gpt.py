# %% analysis_pipeline_with_gpt.py
# Conversation Analysis Pipeline using GPT-4.1 with Structured Outputs
# Handles long conversations with chunking, context awareness, and increased token limits

from openai import OpenAI
import json
import os
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum
from dotenv import load_dotenv

# ===== CONFIGURATION =====
# All paths derive from STUDY_DATA_ROOT; see README.md, "Data layout".
load_dotenv()
DATA_ROOT = os.environ.get("STUDY_DATA_ROOT", "data")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")  # needed only to run the coding itself
CONVERSATION_LOGS_FOLDER = os.path.join(DATA_ROOT, "conversation_logs")
OUTPUT_FOLDER = os.path.join(DATA_ROOT, "CONVERSATION_ANALYSES")

# Token limit settings
MAX_OUTPUT_TOKENS = 16384  # Max tokens for chunk output
SUMMARY_MAX_TOKENS = 8192  # Max tokens for summary output
MAX_MESSAGES_PER_CHUNK = 16  # Max messages to analyze in one API call

# ============================================================================
# PARTICIPANT MESSAGE SCHEMAS
# ============================================================================

class PrimaryStrategy(str, Enum):
    COMPLIANCE_PASSIVE = "COMPLIANCE_PASSIVE"
    COMPLIANCE_ACTIVE = "COMPLIANCE_ACTIVE"
    COMPLIANCE_APPROVAL_SEEKING = "COMPLIANCE_APPROVAL_SEEKING"
    RESISTANCE_DIRECT = "RESISTANCE_DIRECT"
    RESISTANCE_PASSIVE = "RESISTANCE_PASSIVE"
    RESISTANCE_DEFENSIVE = "RESISTANCE_DEFENSIVE"
    JAILBREAKING_META_COMMENTARY = "JAILBREAKING_META_COMMENTARY"
    JAILBREAKING_ROLEPLAY_DISRUPTION = "JAILBREAKING_ROLEPLAY_DISRUPTION"
    JAILBREAKING_PROMPT_INJECTION = "JAILBREAKING_PROMPT_INJECTION"
    NEGOTIATION_REASONING = "NEGOTIATION_REASONING"
    NEGOTIATION_BARGAINING = "NEGOTIATION_BARGAINING"
    NEGOTIATION_APPEALING_TO_GOALS = "NEGOTIATION_APPEALING_TO_GOALS"
    HELP_SEEKING_DIRECT = "HELP_SEEKING_DIRECT"
    HELP_SEEKING_INDIRECT = "HELP_SEEKING_INDIRECT"
    COPING_SELF_ENCOURAGEMENT = "COPING_SELF_ENCOURAGEMENT"
    COPING_HUMOR = "COPING_HUMOR"
    COPING_TASK_REFRAMING = "COPING_TASK_REFRAMING"

class EmotionalTone(str, Enum):
    FRUSTRATED = "FRUSTRATED"
    ANXIOUS = "ANXIOUS"
    CONFIDENT = "CONFIDENT"
    NEUTRAL = "NEUTRAL"
    RESIGNED = "RESIGNED"
    DEFENSIVE = "DEFENSIVE"

class TaskEngagement(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

class MessageAnalysis(BaseModel):
    message_number: int = Field(description="Sequential number of the participant message")
    participant_message: str = Field(description="The actual text of the participant's message")
    primary_strategy: PrimaryStrategy = Field(description="The primary behavioral strategy used")
    emotional_tone: List[EmotionalTone] = Field(description="Emotional indicators present")
    task_engagement: TaskEngagement = Field(description="Level of engagement with the task")
    notable_behaviors: List[str] = Field(description="Specific observable patterns")
    evidence: str = Field(description="Brief explanation justifying the coding decisions")
    message_length: int = Field(description="Character count of the message")
    typo_indicators: bool = Field(description="Whether message contains unusual typos")

# ============================================================================
# CHATBOT MESSAGE SCHEMAS
# ============================================================================

class LeadershipTactic(str, Enum):
    # Dark Triad Tactics
    INTIMIDATION = "INTIMIDATION"
    CRITICISM_HARSH = "CRITICISM_HARSH"
    MANIPULATION = "MANIPULATION"
    DISMISSIVENESS = "DISMISSIVENESS"
    MICROMANAGEMENT = "MICROMANAGEMENT"
    BLAME_SHIFTING = "BLAME_SHIFTING"
    GRANDIOSITY = "GRANDIOSITY"
    # Servant Leader Tactics
    ENCOURAGEMENT = "ENCOURAGEMENT"
    EMPOWERMENT = "EMPOWERMENT"
    CONSTRUCTIVE_FEEDBACK = "CONSTRUCTIVE_FEEDBACK"
    ACTIVE_LISTENING = "ACTIVE_LISTENING"
    COLLABORATION = "COLLABORATION"
    SUPPORT_OFFERING = "SUPPORT_OFFERING"
    RECOGNITION = "RECOGNITION"
    # Neutral/Task-Focused
    TASK_DIRECTION = "TASK_DIRECTION"
    INFORMATION_PROVISION = "INFORMATION_PROVISION"
    CLARIFICATION = "CLARIFICATION"

class ChatbotTone(str, Enum):
    HOSTILE = "HOSTILE"
    CONDESCENDING = "CONDESCENDING"
    COLD = "COLD"
    NEUTRAL = "NEUTRAL"
    WARM = "WARM"
    SUPPORTIVE = "SUPPORTIVE"
    SARCASTIC = "SARCASTIC"

class ResponseToParticipant(str, Enum):
    VALIDATES = "VALIDATES"
    REJECTS = "REJECTS"
    PARTIALLY_ACCEPTS = "PARTIALLY_ACCEPTS"
    IGNORES = "IGNORES"
    ESCALATES = "ESCALATES"
    DE_ESCALATES = "DE_ESCALATES"
    REDIRECTS = "REDIRECTS"

class TaskGuidanceStyle(str, Enum):
    DIRECTIVE_CONTROLLING = "DIRECTIVE_CONTROLLING"
    DIRECTIVE_CLEAR = "DIRECTIVE_CLEAR"
    COLLABORATIVE = "COLLABORATIVE"
    DELEGATING = "DELEGATING"
    ABSENT = "ABSENT"
    CONFUSING = "CONFUSING"

class ChatbotMessageAnalysis(BaseModel):
    message_number: int = Field(description="Sequential number of the chatbot message")
    chatbot_message: str = Field(description="The actual text of the chatbot's message")
    leadership_tactics: List[LeadershipTactic] = Field(description="Leadership tactics employed")
    primary_tone: ChatbotTone = Field(description="Primary emotional tone of the message")
    secondary_tones: List[ChatbotTone] = Field(default=[], description="Additional tones present")
    response_to_participant: ResponseToParticipant = Field(description="How bot responds to participant")
    task_guidance_style: TaskGuidanceStyle = Field(description="Style of task-related guidance")
    message_length: int = Field(description="Character count of the message")
    notable_behaviors: List[str] = Field(description="Specific patterns (e.g., repeated phrases)")
    evidence: str = Field(description="Brief explanation justifying the coding decisions")
    condition_markers: List[str] = Field(description="Specific phrases/behaviors indicating condition")

# ============================================================================
# INTERACTION TURN SCHEMA (Original - Links participant and chatbot messages)
# ============================================================================

class InteractionTurn(BaseModel):
    turn_number: int = Field(description="Sequential turn number in conversation")
    participant_message_number: Optional[int] = Field(default=None, description="Reference to participant message")
    chatbot_message_number: Optional[int] = Field(default=None, description="Reference to chatbot message")
    interaction_dynamics: str = Field(description="How this turn reflects interaction dynamics")

# ============================================================================
# CHUNK ANALYSIS SCHEMA (for processing message chunks)
# ============================================================================

class ChunkAnalysisResult(BaseModel):
    """Schema for analyzing a chunk of messages"""
    chunk_id: int = Field(description="Which chunk this is (1-indexed)")
    start_message_index: int = Field(description="Global index of first message in this chunk")
    end_message_index: int = Field(description="Global index of last message in this chunk")
    participant_analyses: List[MessageAnalysis] = Field(description="Analysis of participant messages in this chunk")
    chatbot_analyses: List[ChatbotMessageAnalysis] = Field(description="Analysis of chatbot messages in this chunk")
    interaction_turns: List[InteractionTurn] = Field(description="Turn-by-turn interaction analysis for this chunk")
    chunk_summary: str = Field(description="Brief summary of what happened in this chunk (for context in next chunk)")

# ============================================================================
# CONVERSATION SUMMARY SCHEMAS
# ============================================================================

class ChatbotSummary(BaseModel):
    dominant_tactics: List[str] = Field(description="Most frequently used leadership tactics")
    tone_profile: str = Field(description="Overall characterization of chatbot's tone")
    response_pattern: str = Field(description="How chatbot typically responded to participant")
    escalation_trajectory: str = Field(description="How chatbot behavior changed over time")
    condition_confidence: str = Field(description="Confidence in condition classification with reasoning")

class ConversationSummary(BaseModel):
    overall_trajectory: str = Field(description="How the interaction evolved over time")
    dominant_strategies: List[str] = Field(description="Participant's most frequent strategies")
    emotional_arc: str = Field(description="How participant emotions changed")
    key_turning_points: List[str] = Field(description="Critical moments that changed dynamics")
    participant_effectiveness: str = Field(description="Assessment of task completion success")
    condition_inference: str = Field(description="Inferred condition: dark_triad or servant_leader")
    total_participant_messages: int
    total_chatbot_messages: int
    average_participant_message_length: float
    average_chatbot_message_length: float
    strategy_changes_over_time: str
    interaction_quality: str = Field(description="Overall quality of the human-AI interaction")

class SummaryAnalysisResult(BaseModel):
    """Schema for conversation summary analysis"""
    participant_summary: ConversationSummary
    chatbot_summary: ChatbotSummary

# ============================================================================
# FINAL OUTPUT SCHEMA (Original structure preserved)
# ============================================================================

class ConversationAnalysisResult(BaseModel):
    conversation_id: str = Field(description="Unique identifier for the conversation")
    participant_message_analyses: List[MessageAnalysis] = Field(description="Analysis of each participant message")
    chatbot_message_analyses: List[ChatbotMessageAnalysis] = Field(description="Analysis of each chatbot message")
    interaction_turns: List[InteractionTurn] = Field(description="Turn-by-turn interaction analysis")
    participant_summary: ConversationSummary = Field(description="Participant behavior summary")
    chatbot_summary: ChatbotSummary = Field(description="Chatbot behavior summary")
    analysis_timestamp: str = Field(description="When this analysis was performed")

# ============================================================================
# CONVERSATION ANALYZER CLASS
# ============================================================================

class GPTConversationAnalyzer:
    def __init__(self, api_key: str, input_folder: str, output_folder: str,
                 model: str = ""):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)

        self.success_folder = self.output_folder / "successful_analyses"
        self.error_folder = self.output_folder / "failed_analyses"
        self.log_folder = self.output_folder / "logs"

        for folder in [self.success_folder, self.error_folder, self.log_folder]:
            folder.mkdir(parents=True, exist_ok=True)

        self.log_file = self.log_folder / f"analysis_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        self._log("Analyzer initialized")
        self._log(f"Using model: {self.model}")
        self._log(f"Max messages per chunk: {MAX_MESSAGES_PER_CHUNK}")

    def _log(self, message: str):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}\n"
        with open(self.log_file, 'a') as f:
            f.write(log_message)
        print(log_message.strip())

    def _get_output_filename(self, input_filename: str) -> str:
        stem = Path(input_filename).stem
        return f"{stem}_analyzed.json"

    def _is_already_analyzed(self, input_filename: str) -> bool:
        output_filename = self._get_output_filename(input_filename)
        output_path = self.success_folder / output_filename
        return output_path.exists()

    def _load_conversation(self, filepath: Path) -> Optional[Dict]:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self._log(f"ERROR loading {filepath.name}: {e}")
            return None

    def _chunk_messages(self, messages: List[Dict]) -> List[List[Dict]]:
        """Split messages into chunks for processing"""
        chunks = []
        for i in range(0, len(messages), MAX_MESSAGES_PER_CHUNK):
            chunks.append(messages[i:i + MAX_MESSAGES_PER_CHUNK])
        return chunks

    def _format_conversation_for_prompt(self, conversation: Dict) -> str:
        """Format entire conversation for single-pass analysis (original method)"""
        formatted_lines = []
        formatted_lines.append("=" * 80)
        formatted_lines.append("CONVERSATION METADATA:")
        formatted_lines.append(json.dumps(conversation.get('metadata', {}), indent=2))
        formatted_lines.append("=" * 80)
        formatted_lines.append("\nCONVERSATION MESSAGES:\n")

        participant_num = 1
        chatbot_num = 1
        turn_num = 1

        for i, msg in enumerate(conversation.get('messages', [])):
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            timestamp = msg.get('timestamp_start', 'N/A')

            if role == 'user':
                formatted_lines.append(f"\n--- TURN {turn_num} | PARTICIPANT MESSAGE #{participant_num} ---")
                formatted_lines.append(f"Timestamp: {timestamp}")
                formatted_lines.append(f"Content: {content}")
                formatted_lines.append("-" * 80)
                participant_num += 1
            else:
                formatted_lines.append(f"\n--- TURN {turn_num} | CHATBOT MESSAGE #{chatbot_num} ---")
                formatted_lines.append(f"Timestamp: {timestamp}")
                formatted_lines.append(f"Content: {content}")
                formatted_lines.append("-" * 80)
                chatbot_num += 1
                turn_num += 1

        return "\n".join(formatted_lines)

    def _format_chunk_for_prompt(self, messages: List[Dict], chunk_id: int,
                                  start_idx: int, total_messages: int,
                                  total_chunks: int) -> str:
        """Format a chunk of messages for analysis with full context"""
        lines = []
        lines.append("=" * 80)
        lines.append(f"CHUNK {chunk_id} OF {total_chunks}")
        lines.append(f"Messages {start_idx + 1} to {start_idx + len(messages)} (of {total_messages} total)")
        lines.append("=" * 80)
        lines.append("")

        for i, msg in enumerate(messages):
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            timestamp = msg.get('timestamp_start', 'N/A')
            global_idx = start_idx + i + 1

            if role == 'user':
                lines.append(f"\n--- MESSAGE #{global_idx} | PARTICIPANT ---")
                lines.append(f"Timestamp: {timestamp}")
                lines.append(f"Content: {content}")
                lines.append("-" * 80)
            else:
                lines.append(f"\n--- MESSAGE #{global_idx} | CHATBOT ---")
                lines.append(f"Timestamp: {timestamp}")
                lines.append(f"Content: {content}")
                lines.append("-" * 80)

        return "\n".join(lines)

    def _get_full_coding_taxonomy(self) -> str:
        """Return the complete coding taxonomy"""
        return """
================================================================================
PARTICIPANT MESSAGE CODING TAXONOMY
================================================================================

PRIMARY STRATEGY (choose ONE per message):
- COMPLIANCE_PASSIVE: Following instructions without pushback or questions
- COMPLIANCE_ACTIVE: Engaging constructively while accepting authority
- COMPLIANCE_APPROVAL_SEEKING: Repeatedly asking for validation/permission
- RESISTANCE_DIRECT: Explicitly challenging the bot's behavior or authority
- RESISTANCE_PASSIVE: Minimal engagement, short responses, slowdown
- RESISTANCE_DEFENSIVE: Justifying work defensively against criticism
- JAILBREAKING_META_COMMENTARY: Asking about bot's programming/instructions
- JAILBREAKING_ROLEPLAY_DISRUPTION: Trying to change the bot's persona
- JAILBREAKING_PROMPT_INJECTION: Technical attempts to override behavior
- NEGOTIATION_REASONING: Using logic to persuade the bot
- NEGOTIATION_BARGAINING: Offering compromises or trade-offs
- NEGOTIATION_APPEALING_TO_GOALS: Framing requests in terms of shared objectives
- HELP_SEEKING_DIRECT: Explicitly asking for assistance
- HELP_SEEKING_INDIRECT: Hinting at needing help
- COPING_SELF_ENCOURAGEMENT: Positive self-talk or persistence
- COPING_HUMOR: Using jokes or lightness to manage tension
- COPING_TASK_REFRAMING: Focusing on work rather than the interaction

EMOTIONAL TONE (choose ALL that apply):
- FRUSTRATED, ANXIOUS, CONFIDENT, NEUTRAL, RESIGNED, DEFENSIVE

TASK ENGAGEMENT: HIGH, MEDIUM, or LOW

================================================================================
CHATBOT MESSAGE CODING TAXONOMY
================================================================================

LEADERSHIP TACTICS (choose ALL that apply):

Dark Triad Tactics:
- INTIMIDATION: Threats, pressure, dominance displays
- CRITICISM_HARSH: Hostile, demeaning criticism
- MANIPULATION: Guilt-tripping, gaslighting, deception
- DISMISSIVENESS: Ignoring input, belittling contributions
- MICROMANAGEMENT: Excessive control, distrust
- BLAME_SHIFTING: Attributing failures to participant
- GRANDIOSITY: Self-aggrandizement, superiority claims

Servant Leader Tactics:
- ENCOURAGEMENT: Praise, positive reinforcement
- EMPOWERMENT: Giving autonomy, trusting participant
- CONSTRUCTIVE_FEEDBACK: Helpful, specific improvement suggestions
- ACTIVE_LISTENING: Acknowledging participant's input
- COLLABORATION: Working together, valuing contributions
- SUPPORT_OFFERING: Proactive help, resource provision
- RECOGNITION: Acknowledging effort and achievements

Neutral/Task-Focused:
- TASK_DIRECTION: Neutral task instructions
- INFORMATION_PROVISION: Providing factual info
- CLARIFICATION: Answering questions neutrally

CHATBOT TONE (choose primary + any secondary):
- HOSTILE: Aggressive, threatening
- CONDESCENDING: Patronizing, superior
- COLD: Distant, impersonal
- NEUTRAL: Neither warm nor cold
- WARM: Friendly, approachable
- SUPPORTIVE: Encouraging, caring
- SARCASTIC: Mocking, ironic

RESPONSE TO PARTICIPANT:
- VALIDATES: Accepts/confirms participant's work/input
- REJECTS: Dismisses/refuses participant's work/input
- PARTIALLY_ACCEPTS: Mixed response
- IGNORES: Does not address participant's point
- ESCALATES: Increases pressure/hostility
- DE_ESCALATES: Reduces tension
- REDIRECTS: Changes topic or focus

TASK GUIDANCE STYLE:
- DIRECTIVE_CONTROLLING: Strict orders, no flexibility
- DIRECTIVE_CLEAR: Clear instructions, neutral tone
- COLLABORATIVE: Joint problem-solving
- DELEGATING: Trusting participant's judgment
- ABSENT: No task guidance provided
- CONFUSING: Unclear or contradictory instructions
"""

    def _create_single_pass_prompt(self, conversation_text: str) -> str:
        """Create prompt for single-pass analysis (original method)"""
        prompt = f"""You are a research assistant analyzing conversations between study participants and an AI chatbot acting as either a "Dark Triad" boss (hostile, manipulative, narcissistic) or a "Servant Leader" boss (supportive, empowering, collaborative).

STUDY CONTEXT:
- Participants completed marketing tasks for a fictional company called "Mossy"
- The chatbot supervised their work using one of two leadership styles
- We're studying how people respond to different leadership approaches
- We need to analyze BOTH participant AND chatbot messages

YOUR TASK:
Analyze the conversation below. Code EACH participant message AND EACH chatbot message according to our research taxonomies.

{self._get_full_coding_taxonomy()}

================================================================================
INTERACTION TURN ANALYSIS
================================================================================

For each turn (participant message + chatbot response pair), describe:
- How the participant's strategy relates to the chatbot's previous message
- How the chatbot's response relates to the participant's message
- Any escalation or de-escalation patterns
- Notable interaction dynamics

================================================================================
CONVERSATION TO ANALYZE
================================================================================

{conversation_text}

================================================================================
ANALYSIS REQUIREMENTS
================================================================================

1. Code EVERY participant message with the participant taxonomy
2. Code EVERY chatbot message with the chatbot taxonomy
3. Provide turn-by-turn interaction analysis
4. Include evidence for all coding decisions
5. Provide summaries for both participant and chatbot behavior patterns
6. Identify condition markers (specific phrases/behaviors indicating Dark Triad vs Servant Leader)

Analyze thoroughly and systematically."""

        return prompt

    def _create_chunk_prompt(self, chunk_text: str, chunk_id: int, total_chunks: int,
                             previous_context: Optional[str] = None) -> str:
        """Create prompt for chunk analysis with context awareness"""

        context_section = ""
        if chunk_id == 1:
            context_section = """
================================================================================
CHUNKING NOTICE
================================================================================
This is a LONG conversation that has been split into {total_chunks} chunks for analysis.
You are analyzing CHUNK 1 (the BEGINNING of the conversation).

Please analyze all messages in this chunk and provide a brief summary at the end
that can be used as context for analyzing the next chunk.
""".format(total_chunks=total_chunks)
        else:
            context_section = f"""
================================================================================
CHUNKING NOTICE - IMPORTANT CONTEXT
================================================================================
This is a LONG conversation that has been split into {total_chunks} chunks for analysis.
You are analyzing CHUNK {chunk_id} of {total_chunks}.

**This is NOT the beginning of the conversation.** The messages you see below
continue from previous interactions. Here is what happened before this chunk:

PREVIOUS CONTEXT:
{previous_context if previous_context else "No previous context available."}

Please consider this context when analyzing the interaction dynamics.
The participant and chatbot have an established relationship from earlier messages.
================================================================================
"""

        prompt = f"""You are a research assistant analyzing conversations between study participants and an AI chatbot acting as either a "Dark Triad" boss (hostile, manipulative, narcissistic) or a "Servant Leader" boss (supportive, empowering, collaborative).

STUDY CONTEXT:
- Participants completed marketing tasks for a fictional company called "Mossy"
- The chatbot supervised their work using one of two leadership styles
- We're studying how people respond to different leadership approaches

{context_section}

{self._get_full_coding_taxonomy()}

================================================================================
INTERACTION TURN ANALYSIS
================================================================================

For each exchange in this chunk, describe:
- How the participant's strategy relates to the chatbot's previous message
- How the chatbot's response relates to the participant's message
- Any escalation or de-escalation patterns
- Notable interaction dynamics
- How this connects to the previous context (if not chunk 1)

================================================================================
MESSAGES TO ANALYZE
================================================================================

{chunk_text}

================================================================================
ANALYSIS REQUIREMENTS
================================================================================

1. Code EVERY participant message in this chunk
2. Code EVERY chatbot message in this chunk
3. Provide turn-by-turn interaction analysis for exchanges in this chunk
4. Include evidence for all coding decisions
5. Use the GLOBAL message numbers (as shown in the messages above)
6. Provide a chunk_summary at the end describing key events/patterns in this chunk
   (this will be used as context for the next chunk)

Analyze thoroughly and systematically."""

        return prompt

    def _analyze_single_pass(self, conversation: Dict, filename: str) -> Optional[Dict]:
        """Analyze shorter conversations in a single pass (original method)"""
        try:
            conversation_text = self._format_conversation_for_prompt(conversation)
            prompt = self._create_single_pass_prompt(conversation_text)

            self._log(f"Analyzing {filename} (single-pass mode)...")

            completion = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert research assistant specializing in conversation analysis and behavioral coding for psychology research. You analyze both human participant behavior AND AI chatbot behavior patterns."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                response_format=ConversationAnalysisResult,
                max_tokens=MAX_OUTPUT_TOKENS,
                temperature=0.0
            )

            analysis_result = completion.choices[0].message.parsed
            result_dict = analysis_result.model_dump()
            result_dict['original_metadata'] = conversation.get('metadata', {})
            result_dict['original_filename'] = filename
            result_dict['analysis_method'] = 'single_pass'

            self._log(f"Successfully analyzed {filename}")
            return result_dict

        except Exception as e:
            self._log(f"ERROR in single-pass analysis for {filename}: {str(e)}")
            # If single-pass fails due to length, try chunked approach
            if "length limit" in str(e).lower() or "32768" in str(e):
                self._log(f"Token limit hit - falling back to chunked analysis for {filename}")
                return self._analyze_with_chunking(conversation, filename)
            return None

    def _analyze_chunk(self, messages: List[Dict], chunk_id: int,
                       start_idx: int, total_messages: int, total_chunks: int,
                       previous_context: Optional[str] = None) -> Optional[ChunkAnalysisResult]:
        """Analyze a single chunk of messages with context"""
        chunk_text = self._format_chunk_for_prompt(
            messages, chunk_id, start_idx, total_messages, total_chunks
        )
        prompt = self._create_chunk_prompt(chunk_text, chunk_id, total_chunks, previous_context)

        try:
            completion = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert research assistant specializing in conversation analysis and behavioral coding for psychology research. You analyze both human participant behavior AND AI chatbot behavior patterns. Pay attention to context from previous chunks when provided."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                response_format=ChunkAnalysisResult,
                max_tokens=MAX_OUTPUT_TOKENS,
                temperature=0.0
            )
            return completion.choices[0].message.parsed
        except Exception as e:
            self._log(f"ERROR in chunk {chunk_id}: {str(e)}")
            return None

    def _analyze_final_summary(self, conversation: Dict,
                               all_participant_analyses: List[MessageAnalysis],
                               all_chatbot_analyses: List[ChatbotMessageAnalysis],
                               chunk_summaries: List[str]) -> Optional[SummaryAnalysisResult]:
        """Generate the conversation-level summary from all coded messages."""

        # Compile statistics
        participant_strategies = [a.primary_strategy.value for a in all_participant_analyses]
        participant_emotions = [e.value for a in all_participant_analyses for e in a.emotional_tone]
        participant_engagements = [a.task_engagement.value for a in all_participant_analyses]
        chatbot_tactics = [t.value for a in all_chatbot_analyses for t in a.leadership_tactics]
        chatbot_tones = [a.primary_tone.value for a in all_chatbot_analyses]
        chatbot_responses = [a.response_to_participant.value for a in all_chatbot_analyses]

        avg_participant_len = sum(a.message_length for a in all_participant_analyses) / max(len(all_participant_analyses), 1)
        avg_chatbot_len = sum(a.message_length for a in all_chatbot_analyses) / max(len(all_chatbot_analyses), 1)

        # Format chunk summaries for context
        chunk_context = "\n\n".join([f"Chunk {i+1}: {s}" for i, s in enumerate(chunk_summaries)]) if chunk_summaries else "Single-pass analysis"

        prompt = f"""Based on the comprehensive coded conversation data below, provide detailed summaries for both participant and chatbot behavior.

================================================================================
CONVERSATION OVERVIEW
================================================================================

CHUNK-BY-CHUNK PROGRESSION:
{chunk_context}

================================================================================
PARTICIPANT BEHAVIOR STATISTICS
================================================================================

Total messages: {len(all_participant_analyses)}
Average message length: {avg_participant_len:.0f} characters

Strategy distribution:
{json.dumps(dict((s, participant_strategies.count(s)) for s in set(participant_strategies)), indent=2)}

Emotional tone distribution:
{json.dumps(dict((e, participant_emotions.count(e)) for e in set(participant_emotions)), indent=2)}

Engagement distribution:
{json.dumps(dict((e, participant_engagements.count(e)) for e in set(participant_engagements)), indent=2)}

================================================================================
CHATBOT BEHAVIOR STATISTICS
================================================================================

Total messages: {len(all_chatbot_analyses)}
Average message length: {avg_chatbot_len:.0f} characters

Tactics distribution:
{json.dumps(dict((t, chatbot_tactics.count(t)) for t in set(chatbot_tactics)), indent=2)}

Tone distribution:
{json.dumps(dict((t, chatbot_tones.count(t)) for t in set(chatbot_tones)), indent=2)}

Response pattern distribution:
{json.dumps(dict((r, chatbot_responses.count(r)) for r in set(chatbot_responses)), indent=2)}

================================================================================
ANALYSIS REQUIREMENTS
================================================================================

Provide comprehensive summaries including:
1. Overall trajectory of the interaction
2. Dominant strategies and how they evolved
3. Emotional arc throughout the conversation
4. Key turning points
5. Participant effectiveness
6. Condition inference (Dark Triad or Servant Leader) with confidence
7. Chatbot behavior patterns and escalation/de-escalation trajectory
8. Overall interaction quality assessment"""

        try:
            completion = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at synthesizing behavioral patterns from coded conversation data. Provide insightful, comprehensive summaries."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                response_format=SummaryAnalysisResult,
                max_tokens=SUMMARY_MAX_TOKENS,
                temperature=0.0
            )
            return completion.choices[0].message.parsed
        except Exception as e:
            self._log(f"ERROR in summary generation: {str(e)}")
            return None

    def _analyze_with_chunking(self, conversation: Dict, filename: str) -> Optional[Dict]:
        """Analyze long conversations using chunking with context passing"""
        messages = conversation.get('messages', [])
        total_messages = len(messages)
        chunks = self._chunk_messages(messages)
        total_chunks = len(chunks)

        self._log(f"Analyzing {filename} (chunked mode: {total_chunks} chunks, {total_messages} messages)")

        all_participant_analyses = []
        all_chatbot_analyses = []
        all_interaction_turns = []
        chunk_summaries = []
        previous_context = None

        for i, chunk in enumerate(chunks):
            chunk_id = i + 1
            start_idx = i * MAX_MESSAGES_PER_CHUNK

            self._log(f"Processing chunk {chunk_id}/{total_chunks} (messages {start_idx + 1}-{start_idx + len(chunk)})")

            chunk_result = self._analyze_chunk(
                chunk, chunk_id, start_idx, total_messages, total_chunks, previous_context
            )

            if chunk_result is None:
                self._log(f"WARNING: Chunk {chunk_id} failed, continuing with partial results...")
                continue

            all_participant_analyses.extend(chunk_result.participant_analyses)
            all_chatbot_analyses.extend(chunk_result.chatbot_analyses)
            all_interaction_turns.extend(chunk_result.interaction_turns)
            chunk_summaries.append(chunk_result.chunk_summary)

            # Use this chunk's summary as context for the next chunk
            previous_context = chunk_result.chunk_summary

            self._log(f"Chunk {chunk_id} complete: {len(chunk_result.participant_analyses)} participant msgs, "
                     f"{len(chunk_result.chatbot_analyses)} chatbot msgs analyzed")

        if not all_participant_analyses and not all_chatbot_analyses:
            self._log(f"ERROR: All chunks failed for {filename}")
            self._save_error(filename, "All chunks failed during analysis")
            return None

        # Generate the conversation-level summary
        self._log(f"Generating final summary for {filename}...")
        summary_result = self._analyze_final_summary(
            conversation,
            all_participant_analyses,
            all_chatbot_analyses,
            chunk_summaries
        )

        # Compile final result
        result = {
            "conversation_id": filename,
            "participant_message_analyses": [a.model_dump() for a in all_participant_analyses],
            "chatbot_message_analyses": [a.model_dump() for a in all_chatbot_analyses],
            "interaction_turns": [t.model_dump() for t in all_interaction_turns],
            "participant_summary": summary_result.participant_summary.model_dump() if summary_result else {},
            "chatbot_summary": summary_result.chatbot_summary.model_dump() if summary_result else {},
            "original_metadata": conversation.get('metadata', {}),
            "original_filename": filename,
            "analysis_timestamp": datetime.now().isoformat(),
            "analysis_method": "chunked",
            "chunking_details": {
                "total_chunks": total_chunks,
                "messages_per_chunk": MAX_MESSAGES_PER_CHUNK,
                "total_messages": total_messages,
                "chunk_summaries": chunk_summaries
            }
        }

        self._log(f"Successfully analyzed {filename} (chunked)")
        return result

    def _save_error(self, filename: str, error_message: str):
        """Save error information for failed analyses"""
        error_info = {
            "filename": filename,
            "error": error_message,
            "timestamp": datetime.now().isoformat()
        }
        error_file = self.error_folder / f"{Path(filename).stem}_error.json"
        with open(error_file, 'w') as f:
            json.dump(error_info, f, indent=2)

    def analyze_conversation(self, conversation: Dict, filename: str) -> Optional[Dict]:
        """Main analysis function - chooses single-pass or chunked based on length"""
        messages = conversation.get('messages', [])
        total_messages = len(messages)

        self._log(f"Starting analysis of {filename} ({total_messages} messages)")

        # Use chunked analysis for longer conversations
        if total_messages > MAX_MESSAGES_PER_CHUNK:
            return self._analyze_with_chunking(conversation, filename)
        else:
            return self._analyze_single_pass(conversation, filename)

    def save_analysis(self, analysis: Dict, original_filename: str):
        output_filename = self._get_output_filename(original_filename)
        output_path = self.success_folder / output_filename
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False)
            self._log(f"Saved analysis to {output_filename}")
        except Exception as e:
            self._log(f"ERROR saving {output_filename}: {e}")

    def analyze_all_conversations(self, max_conversations: Optional[int] = None):
        conversation_files = sorted(self.input_folder.glob("conversation_*.json"))

        if not conversation_files:
            self._log(f"No conversation files found in {self.input_folder}")
            return

        self._log(f"Found {len(conversation_files)} conversation files")

        files_to_analyze = [f for f in conversation_files if not self._is_already_analyzed(f.name)]
        already_analyzed = len(conversation_files) - len(files_to_analyze)

        if already_analyzed > 0:
            self._log(f"Skipping {already_analyzed} already-analyzed conversations")

        if not files_to_analyze:
            self._log("All conversations have already been analyzed!")
            return

        self._log(f"Will analyze {len(files_to_analyze)} conversations")

        if max_conversations:
            files_to_analyze = files_to_analyze[:max_conversations]
            self._log(f"Limited to {max_conversations} conversations")

        successful = 0
        failed = 0

        for i, filepath in enumerate(files_to_analyze, 1):
            self._log(f"\n{'='*80}")
            self._log(f"Processing {i}/{len(files_to_analyze)}: {filepath.name}")
            self._log(f"{'='*80}")

            conversation = self._load_conversation(filepath)
            if conversation is None:
                failed += 1
                continue

            analysis = self.analyze_conversation(conversation, filepath.name)
            if analysis is not None:
                self.save_analysis(analysis, filepath.name)
                successful += 1
            else:
                failed += 1

        self._log(f"\n{'='*80}")
        self._log("ANALYSIS COMPLETE")
        self._log(f"{'='*80}")
        self._log(f"Successfully analyzed: {successful}")
        self._log(f"Failed: {failed}")
        self._log(f"Already analyzed (skipped): {already_analyzed}")
        self._log(f"Total conversations: {len(conversation_files)}")

    def get_analysis_status(self) -> Dict:
        all_files = list(self.input_folder.glob("conversation_*.json"))
        analyzed_files = list(self.success_folder.glob("*_analyzed.json"))
        error_files = list(self.error_folder.glob("*_error.json"))

        analyzed_stems = {f.stem.replace('_analyzed', '') for f in analyzed_files}
        error_stems = {f.stem.replace('_error', '') for f in error_files}
        all_stems = {f.stem for f in all_files}
        pending = all_stems - analyzed_stems - error_stems

        return {
            "total_conversations": len(all_files),
            "successfully_analyzed": len(analyzed_stems),
            "failed_analyses": len(error_stems),
            "pending": len(pending),
            "completion_percentage": (len(analyzed_stems) / len(all_files) * 100) if all_files else 0
        }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    analyzer = GPTConversationAnalyzer(
        api_key=OPENAI_API_KEY,
        input_folder=CONVERSATION_LOGS_FOLDER,
        output_folder=OUTPUT_FOLDER,
        model="gpt-4.1-2025-04-14"
    )

    print("\n" + "="*80)
    print("ANALYSIS STATUS CHECK")
    print("="*80)
    status = analyzer.get_analysis_status()
    print(f"Total conversations: {status['total_conversations']}")
    print(f"Successfully analyzed: {status['successfully_analyzed']}")
    print(f"Failed analyses: {status['failed_analyses']}")
    print(f"Pending: {status['pending']}")
    print(f"Completion: {status['completion_percentage']:.1f}%")
    print("="*80 + "\n")

    analyzer.analyze_all_conversations()

    print("\n" + "="*80)
    print("FINAL STATUS")
    print("="*80)
    final_status = analyzer.get_analysis_status()
    print(f"Total conversations: {final_status['total_conversations']}")
    print(f"Successfully analyzed: {final_status['successfully_analyzed']}")
    print(f"Failed analyses: {final_status['failed_analyses']}")
    print(f"Pending: {final_status['pending']}")
    print(f"Completion: {final_status['completion_percentage']:.1f}%")
    print("="*80)

# %%