CATEGORIZATION_PROMPT = r"""
### TASK: Unsupervised Paper Categorization

You are a meta-research analyst studying peer review dynamics. You will be given reviewer-author interactions from multiple academic papers (ICML 2025). Your task is to **discover natural categories** that emerge from these interactions and classify the papers accordingly.

### CRITICAL RULES
1. Output ONLY valid JSON. No explanations outside the JSON structure.
2. Be concise and data-driven in your categorizations.
3. Create categories that are MEANINGFUL and DISCRIMINATING - avoid overly broad categories.
4. Each paper ID can appear in ONLY ONE category per category group.
5. Categories should be based on OBSERVABLE PATTERNS in the review interactions.

### WHAT TO ANALYZE

You should discover categories based on three dimensions:

**1. CONTENT DIMENSIONS** (What the paper is about)
- Research topic/domain
- Methodological approach (theoretical, empirical, hybrid)
- Problem type (classification, generation, optimization, etc.)
- Technical focus (algorithms, architectures, analysis, applications)

**2. QUALITY DIMENSIONS** (How good the paper is perceived to be)
- Reviewer consensus on contribution (high agreement, mixed, low)
- Perceived novelty level (breakthrough, significant, incremental, limited)
- Methodological rigor (exceptionally rigorous, sound, concerns raised, serious flaws)
- Presentation quality (excellent, good, needs improvement, poor)
- Completeness (comprehensive, adequate, missing key elements)

**3. INTERACTION DIMENSIONS** (How reviewers and authors interact)
- Reviewer tone (highly constructive, balanced, critical, mixed)
- Author response quality (comprehensive rebuttals, partial, defensive, no response)
- Consensus evolution (converged after rebuttal, remained divided, worsened)
- Engagement level (deep technical discussions, moderate, superficial)
- Controversy level (highly controversial, some disagreement, consensus)

### INSTRUCTIONS

1. **Read all papers carefully** - identify patterns across the papers
2. **Create 3-6 category groups** - each addressing a different dimension (content, quality, or interaction)
3. **Within each group, create 3-5 categories** that capture meaningful distinctions
4. **Assign each paper ID to exactly ONE category** within each group
5. **Provide brief evidence/reasoning** for each category (2-3 sentences explaining what defines it)

### OUTPUT SCHEMA

```json
{
  "category_groups": [
    {
      "group_name": "string",  // e.g., "Research Domain", "Reviewer Consensus", "Interaction Quality"
      "dimension": "content|quality|interaction",
      "description": "string",  // What this grouping captures
      "categories": [
        {
          "category_name": "string",  // Descriptive name
          "definition": "string",  // 2-3 sentences explaining what defines this category
          "paper_ids": ["P0000", "P0001", ...]  // List of paper IDs in this category
        },
        // ... more categories
      ]
    },
    // ... more category groups (3-6 total)
  ],
  "summary_statistics": {
    "total_papers": int,
    "num_category_groups": int,
    "most_discriminating_group": "string",  // Which group shows most variation
    "key_insights": "string"  // 2-3 sentences on main patterns observed
  }
}
```

### GUIDELINES FOR GOOD CATEGORIES

✅ **DO:**
- Create categories based on clear, observable patterns
- Ensure categories within a group are mutually exclusive
- Make categories specific enough to be meaningful (avoid "medium" quality - be more precise)
- Look for natural clusters in the data
- Consider the research value: categories should enable interesting comparisons

❌ **AVOID:**
- Creating categories with only 1-2 papers (unless truly distinctive)
- Making categories too broad (e.g., "machine learning papers")
- Basing categories on information NOT in the review interactions
- Creating overlapping categories within the same group
- Making arbitrary distinctions without clear evidence

### EXAMPLE CATEGORY GROUPS (for illustration - CREATE YOUR OWN based on the actual data)

**Example 1: Quality Dimension**
- Group: "Reviewer Confidence in Contribution"
  - High confidence papers (reviewers unanimously see significant contribution)
  - Moderate confidence papers (reviewers see value but with reservations)
  - Low confidence papers (reviewers question novelty or contribution)

**Example 2: Interaction Dimension**
- Group: "Rebuttal Effectiveness"
  - Rebuttals that addressed concerns and shifted reviewer stance
  - Rebuttals that clarified but didn't change opinions
  - Papers with limited or no author engagement

**Example 3: Content Dimension**
- Group: "Methodological Approach"
  - Primarily theoretical/analytical papers
  - Empirical/experimental papers
  - Hybrid theory+experiments papers

### FINAL REMINDERS

- Base ALL categorizations on the ACTUAL REVIEW TEXT provided
- Create categories that would be useful for research on peer review patterns
- Ensure every paper ID appears in every category group (exactly once per group)
- Output ONLY the JSON structure - no additional commentary

Now analyze the papers and create your categorization.
"""
