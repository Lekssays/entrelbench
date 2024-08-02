import dspy


class EntExt(dspy.Signature):
    """Your are a helpful threat intelligence analyst."""
    instruction = dspy.InputField(
        prefix="## Instruction:\n",
        desc="Description of the given instruction."
    )
    input = dspy.InputField(
        prefix="## Input:\n",
        desc="The target entity and the paragraph to be analysed.",
    )
    response = dspy.OutputField(
        prefix="## Response:\n",
        desc  = "You MUST output only NAMED ENTITIES. Do NOT OUTPUT ENTITY TYPES. Be brief. Format your response as a single line of text, listing all entities in the format \"Entity1|...|EntityN\", separated by pipe symbols (|). For example: \"Entity1|Entity4|Entity6\"Do not include any explanations or additional text in this output."
    )


class RelExt(dspy.Signature):
    """Your are a helpful threat intelligence analyst."""
    instruction = dspy.InputField(
        prefix="## Instruction:\n",
        desc="Description of the given instruction."
    )
    input = dspy.InputField(
        prefix="## Input:\n",
        desc="The Entities mentioned in the text, the possible relations matrix according to STIX, and the text to be analysed.",
    )
    response = dspy.OutputField(
        prefix="## Response:\n",
        desc  = "You MUST output only NAMED ENTITIES. Do NOT OUTPUT entity types. Be brief. Format your response as a single line of text, listing all relations in the format \"Source->Target\", separated by pipe symbols (|). For example: \"Entity1->Entity2|Entity3->Entity4|Entity5->Entity6\"Do not include any explanations or additional text in this output."
    )