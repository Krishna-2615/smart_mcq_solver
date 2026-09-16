if os.path.exists("./kuzu_mcq_db"):
    shutil.rmtree("./kuzu_mcq_db")

import kuzu
db = kuzu.Database("./kuzu_mcq_db")
conn = kuzu.Connection(db)

conn.execute("CREATE NODE TABLE Topic(name STRING, PRIMARY KEY(name))")
conn.execute("CREATE NODE TABLE Term(name STRING, definition STRING, PRIMARY KEY(name))")
conn.execute("CREATE REL TABLE HAS_TERM(FROM Topic TO Term)")

seen_terms = set()
for path in txt_files:
    title, terms, _ = parse_structured_txt(path)
    conn.execute("MERGE (t:Topic {name: $name})", parameters={"name": title})
    for term, definition in terms:
        term, definition = term.strip(), definition.strip()
        if term not in seen_terms:
            conn.execute(
                "CREATE (te:Term {name: $name, definition: $definition})",
                parameters={"name": term, "definition": definition},
            )
            seen_terms.add(term)
        conn.execute(
            "MATCH (t:Topic), (te:Term) WHERE t.name = $tname AND te.name = $ename "
            "CREATE (t)-[:HAS_TERM]->(te)",
            parameters={"tname": title, "ename": term},
        )

print(f"Graph: {len(txt_files)} topics, {len(seen_terms)} unique term nodes")

sorted_term_keys = sorted(all_terms.keys(), key=len, reverse=True)

def graph_context(query, max_terms=3):
    q_lower = query.lower()
    matched = [k for k in sorted_term_keys if k in q_lower][:max_terms]
    if not matched:
        return ""
    pieces = []
    for k in matched:
        term, definition, _ = all_terms[k]
        res = conn.execute(
            "MATCH (t:Topic)-[:HAS_TERM]->(te:Term) WHERE te.name = $name "
            "RETURN t.name, te.definition",
            parameters={"name": term},
        )
        while res.has_next():
            topic_name, defn = res.get_next()
            pieces.append(f"[{topic_name}] {term}: {defn}")
    return "\n".join(pieces)

def combined_context(query, max_terms=3, fallback_k=3):
    ctx = graph_context(query, max_terms=max_terms)
    return ctx if ctx else faiss_context(query, k=fallback_k)
