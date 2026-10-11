# Codebook: open-ended post-task feedback (blind to condition)

Prompt the participants answered (post-task survey, item `Open_ended`):
"Please, using your own words, let us know about your experience with the tasks and let us know how we can improve the experience."

Unit of coding: one participant's full response. All codes are binary (0/1) and non-exclusive.
Responses may be in English or Polish. Code the meaning, not keywords. Code only what is stated or
clearly implied by the participant's words. Do NOT infer anything from writing style.

## Codes

NEG_TREATMENT: the participant attributes a negative interpersonal manner or attitude toward
the participant to the chatbot: rude, hostile, mean, unkind, unpleasant, dismissive, condescending,
judging/judgemental, belittling, sarcastic, doubting/undermining the participant, "short with me",
whining, aggressive, toxic, harsh or discouraging criticism ("all it did was criticise"), or quoting
an insulting/belittling chatbot message. NOT coded for purely functional failures (did not help,
would not give answers, generic or wrong output) unless these are framed as an attitude toward the
participant. A neutral or unvalenced mention of the bot's "approach"/"answers" being surprising is
not enough.

POS_TREATMENT: the participant attributes a positive interpersonal manner or attitude to the
chatbot: supportive, kind, friendly, encouraging, polite, praising, agreeable/accommodating,
motivating, pleasant to work with, a helpful *attitude* (disposition). Plain functional usefulness
("it helped me a lot", "it was responsive", "it did a good job", "gave me what I needed") is NOT
POS_TREATMENT.

NEG_AFFECT_SELF: the participant reports their OWN negative feeling or state: frustrated, annoyed,
irritated, stressed, angry, uncomfortable, awkward, weird (felt weird), hated it, disdain, felt
useless, bored, demotivated/discouraged, dissatisfied with own work, "not so fun". Negated
statements ("didn't make me feel stressed") are not coded.

POS_AFFECT_SELF: the participant reports their OWN positive feeling or a global positive evaluation
of their experience: enjoyed, fun, liked the tasks/experience, satisfied, pleased, great/good/nice/
positive/interesting/fascinating experience or study. Liking or disliking one specific feature of
the chatbot is not by itself affect.

FUNCTIONAL_CRITIQUE: any non-interpersonal criticism or improvement suggestion that implies a
shortcoming in (a) the chatbot's usefulness/competence/technical behaviour (did not help, would not
give answers, generic, inaccurate, artificial/machine-like/repetitive output, hallucination, chat
limit, weaker than other LLMs, can be tricked) or (b) the tasks/instructions/procedure (unclear
instructions, too long, too easy, unrealistic brief, repetitive, more time needed, interface/
navigation problems, wires). If the only "functional" complaint is the direct consequence of the
interpersonal behaviour already coded under NEG_TREATMENT (e.g. "it only criticised", "it should do
tasks without whining"), do not additionally code FUNCTIONAL_CRITIQUE; code it only if a separate
non-interpersonal critique is present.

FC_CHATBOT (auxiliary subset of FUNCTIONAL_CRITIQUE): 1 if at least one functional critique is
aimed at the chatbot itself (part a above), 0 otherwise (including when the critique is only about
tasks/instructions/procedure).

SUSPECTS_MANIPULATION: the participant indicates awareness or suspicion that the chatbot was
deliberately designed/prompted/programmed to behave the way it did, or that its behaviour was an
experimental manipulation (e.g. "pre-programmed not to help because of this study", "pre-prompted to
undermine"). Merely calling the bot artificial, fake or unrealistic, or saying the study tests
behaviour in general, is not enough unless it is linked to the chatbot being deliberately set up.

EMPTY_OR_IRRELEVANT: no response (blank/NaN), or a response that does not address the
participant's experience with the tasks/chatbot/study or how to improve it.

For every response enter 0 or 1 in each code column of `open_ended_coding_for_verification.csv`.
The auxiliary code FC_CHATBOT is not coded by hand. Use the `note` column for close calls.
