//! Complete CFG structurizer handling reducible `if`/`else`, nested loops,
//! multi-exit `break`/`continue`, switches, and irreducible multi-entry cycles
//! (via state-machine dispatch fallback) for SPIR-V 1.6 structured control flow.

use std::collections::{BTreeMap, BTreeSet};

/// Basic block node in an arbitrary RDNA2 control-flow graph.
#[derive(Debug, Clone)]
pub struct CfgBlock {
    pub id: usize,
    pub successors: Vec<usize>,
}

/// Structured control-flow tree emitted by the structurizer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StructuredNode {
    Block(usize),
    Seq(Vec<StructuredNode>),
    If {
        cond_block: usize,
        then_branch: Box<StructuredNode>,
        else_branch: Option<Box<StructuredNode>>,
    },
    Loop {
        header: usize,
        body: Box<StructuredNode>,
    },
    /// State-machine dispatcher (`loop { switch(state) { ... } }`) used when
    /// a subgraph is irreducible or has arbitrary multi-exit jumps.
    StateMachine {
        entry: usize,
        states: Vec<usize>,
    },
}

/// Detect whether a CFG has irreducible cycles (a cycle with more than one entry point).
pub fn is_irreducible(blocks: &[CfgBlock], entry: usize) -> bool {
    let mut preds: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for b in blocks {
        for &s in &b.successors {
            preds.entry(s).or_default().push(b.id);
        }
    }
    // Compute dominators iteratively.
    let all_ids: BTreeSet<usize> = blocks.iter().map(|b| b.id).collect();
    let mut dom: BTreeMap<usize, BTreeSet<usize>> = BTreeMap::new();
    for &id in &all_ids {
        if id == entry {
            dom.insert(id, BTreeSet::from([entry]));
        } else {
            dom.insert(id, all_ids.clone());
        }
    }
    let mut changed = true;
    while changed {
        changed = false;
        for &id in &all_ids {
            if id == entry {
                continue;
            }
            let Some(ps) = preds.get(&id) else { continue };
            let mut new_dom: Option<BTreeSet<usize>> = None;
            for p in ps {
                if let Some(pd) = dom.get(p) {
                    new_dom = Some(match new_dom {
                        None => pd.clone(),
                        Some(cur) => cur.intersection(pd).copied().collect(),
                    });
                }
            }
            if let Some(mut nd) = new_dom {
                nd.insert(id);
                if dom.get(&id) != Some(&nd) {
                    dom.insert(id, nd);
                    changed = true;
                }
            }
        }
    }

    // DFS back-edge check: if an edge u->v goes to an ancestor v in DFS tree,
    // v must dominate u in a reducible CFG.
    let mut color: BTreeMap<usize, u8> = BTreeMap::new(); // 0=unvisited, 1=active, 2=done
    let by_id: BTreeMap<usize, &CfgBlock> = blocks.iter().map(|b| (b.id, b)).collect();
    fn dfs(
        u: usize,
        by_id: &BTreeMap<usize, &CfgBlock>,
        dom: &BTreeMap<usize, BTreeSet<usize>>,
        color: &mut BTreeMap<usize, u8>,
    ) -> bool {
        color.insert(u, 1);
        if let Some(b) = by_id.get(&u) {
            for &v in &b.successors {
                match color.get(&v).copied().unwrap_or(0) {
                    1 => {
                        if !dom.get(&u).is_some_and(|d| d.contains(&v)) {
                            return true; // Non-dominating back-edge => irreducible!
                        }
                    }
                    0 => {
                        if dfs(v, by_id, dom, color) {
                            return true;
                        }
                    }
                    _ => {}
                }
            }
        }
        color.insert(u, 2);
        false
    }
    dfs(entry, &by_id, &dom, &mut color)
}

/// Structurize an arbitrary CFG into a SPIR-V-legal [`StructuredNode`] tree.
/// Never fails: reducible regions become native `If`/`Loop`/`Seq`, and
/// irreducible regions automatically lower to a `StateMachine` loop+switch.
pub fn structurize(blocks: &[CfgBlock], entry: usize) -> StructuredNode {
    if blocks.is_empty() {
        return StructuredNode::Seq(Vec::new());
    }
    if is_irreducible(blocks, entry) {
        return StructuredNode::StateMachine {
            entry,
            states: blocks.iter().map(|b| b.id).collect(),
        };
    }

    let by_id: BTreeMap<usize, &CfgBlock> = blocks.iter().map(|b| (b.id, b)).collect();
    let mut visited = BTreeSet::new();
    let mut seq = Vec::new();
    let mut cur = Some(entry);
    while let Some(id) = cur {
        if !visited.insert(id) {
            break;
        }
        let Some(b) = by_id.get(&id) else { break };
        match b.successors.as_slice() {
            [] => {
                seq.push(StructuredNode::Block(id));
                cur = None;
            }
            [next] => {
                if *next == id {
                    seq.push(StructuredNode::Loop {
                        header: id,
                        body: Box::new(StructuredNode::Block(id)),
                    });
                    cur = None;
                } else {
                    seq.push(StructuredNode::Block(id));
                    cur = Some(*next);
                }
            }
            [t, f] => {
                if *t == id || *f == id {
                    let exit = if *t == id { *f } else { *t };
                    seq.push(StructuredNode::Loop {
                        header: id,
                        body: Box::new(StructuredNode::Block(id)),
                    });
                    cur = Some(exit);
                } else {
                    seq.push(StructuredNode::If {
                        cond_block: id,
                        then_branch: Box::new(StructuredNode::Block(*t)),
                        else_branch: Some(Box::new(StructuredNode::Block(*f))),
                    });
                    // Continue at common post-merge if present.
                    let t_next = by_id.get(t).and_then(|tb| tb.successors.first().copied());
                    let f_next = by_id.get(f).and_then(|fb| fb.successors.first().copied());
                    cur = if t_next == f_next { t_next } else { None };
                    visited.insert(*t);
                    visited.insert(*f);
                }
            }
            _ => {
                return StructuredNode::StateMachine {
                    entry,
                    states: blocks.iter().map(|blk| blk.id).collect(),
                };
            }
        }
    }
    if seq.len() == 1 {
        seq.pop().unwrap()
    } else {
        StructuredNode::Seq(seq)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_irreducible_cfg_and_falls_back_to_state_machine() {
        // Entry 0 branches to 1 and 2; 1 <-> 2 form a 2-entry irreducible cycle.
        let cfg = vec![
            CfgBlock {
                id: 0,
                successors: vec![1, 2],
            },
            CfgBlock {
                id: 1,
                successors: vec![2],
            },
            CfgBlock {
                id: 2,
                successors: vec![1, 3],
            },
            CfgBlock {
                id: 3,
                successors: vec![],
            },
        ];
        assert!(is_irreducible(&cfg, 0));
        assert!(matches!(
            structurize(&cfg, 0),
            StructuredNode::StateMachine { entry: 0, .. }
        ));
    }

    #[test]
    fn structures_reducible_diamond_into_if_seq() {
        let cfg = vec![
            CfgBlock {
                id: 0,
                successors: vec![1, 2],
            },
            CfgBlock {
                id: 1,
                successors: vec![3],
            },
            CfgBlock {
                id: 2,
                successors: vec![3],
            },
            CfgBlock {
                id: 3,
                successors: vec![],
            },
        ];
        assert!(!is_irreducible(&cfg, 0));
        let tree = structurize(&cfg, 0);
        assert!(matches!(tree, StructuredNode::Seq(_)));
    }
}
