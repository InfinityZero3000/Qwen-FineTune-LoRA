Đúng hướng là xây paper theo kiểu này:

Không chứng minh “GraphCAG mạnh hơn CAG vì benchmark hơn”.
Chứng minh “GraphCAG là một mở rộng chặt của CAG cho bài toán khó hơn”, rồi benchmark chỉ đóng vai trò xác nhận thực dụng sau đó.

Tức là paper cần có 2 thứ thật rõ:

1. Một lý thuyết:
CAG gốc là trường hợp con của GraphCAG.
2. Một thuật toán:
khi nào giữ exact reuse của CAG, khi nào nâng lên graph-aware reuse, khi nào buộc reconstruct.

Nếu làm được vậy thì paper tự mạnh hơn ở cả theory lẫn practical framing, dù chưa cần nói “hơn bao nhiêu lần”.

**Theory**
Lý thuyết đúng nhất cho bài toán của bạn là:

GraphCAG là một hierarchical reuse theory under state drift.

CAG gốc giải bài toán:
“nếu context còn dùng lại được thì reuse để tránh retrieval/generation cost”.

Bài toán của bạn khó hơn:
“nếu state đã drift, exact reuse không còn đủ; cần một luật để quyết định reuse nào còn an toàn, reuse nào phải sửa cục bộ, reuse nào phải bỏ và reconstruct”.

Vậy lý thuyết trung tâm nên là:

- Có một không gian trạng thái $x = (q, p, s)$.
Here:
- $q$: query / request
- $p$: profile or policy state
- $s$: session or progress state

- Có một reusable artifact $m$ được sinh ở thời điểm trước.
- Ta không hỏi “m có giống text không”.
- Ta hỏi “m còn hợp lệ dưới state hiện tại không”.

Từ đó định nghĩa một quan hệ hợp lệ:

$$
m \models_{\text{PCC}} x
$$

nghĩa là artifact $m$ còn reusable cho trạng thái hiện tại $x$ nếu thỏa PCC.

PCC là phần lý thuyết quan trọng nhất của bạn. Nó không chỉ là heuristic. Nó chính là luật hợp lệ hóa reuse.

Bạn có thể phát biểu thành:

Definition:
A reusable artifact $m$ is admissible for current state $x$ iff it satisfies:
- intent consistency
- concept consistency
- level safety
- progress safety
- freshness

Từ đó suy ra một định lý rất mạnh về mặt positioning:

Theorem 1: Vanilla CAG is a special case of GraphCAG.

Phát biểu kiểu paper:

If the admissibility relation is restricted to exact-match reuse only, the graph-neighborhood candidate set is disabled, and all non-exact cases are forced to reconstruction, then GraphCAG reduces to vanilla CAG.

Viết toán học:

- L1 disabled
- graph bucket set empty
- admissibility = exact key equality

thì:

$$
\text{GraphCAG}(x) = \text{CAG}(x)
$$

Đây là điểm cực mạnh. Nó nói:
GraphCAG không làm yếu CAG.
GraphCAG chứa CAG như trường hợp suy biến L0-only.

Sau đó thêm một định lý nữa, còn mạnh hơn về logic:

Theorem 2: GraphCAG is strictly more expressive than vanilla CAG under state drift.

Ý:
tồn tại các bài toán mà exact reuse của CAG không đủ để mô hình hóa quyết định đúng, nhưng GraphCAG mô hình hóa được nhờ:
- admissibility relation
- graph-local candidate neighborhood
- fallback hierarchy L0/L1/L2

Phát biểu informal:
There exist drift-conditioned request pairs for which exact-match reuse is insufficient, but a graph-aware admissibility controller can still select a safe reusable artifact or correctly reject reuse.

Đây là “strict extension” claim.
Claim này mạnh hơn rất nhiều so với “chúng tôi benchmark tốt hơn”.

**Algorithm**
Thuật toán đúng nhất là giữ RAPID như core algorithm, nhưng phải định nghĩa nó là “hierarchical decision algorithm extending CAG”.

Tên có thể là:

RAPID:
Risk-Aware Progressive Inference Decision

hoặc giữ tên hiện tại của bạn.

Thuật toán nên có đúng 3 mức:

1. L0:
exact reuse, chính là vanilla CAG fast path.
2. L1:
graph-aware near-hit reuse dưới PCC.
3. L2:
grounded reconstruction.

Dạng thuật toán nên nói như sau:

Input:
- current state $x=(q,p,s)$
- exact cache $\mathcal{C}_{L0}$
- graph-local memory cells $\mathcal{C}_{L1}$
- graph $G$

Output:
- decision in $\{L0, L1, L2\}$
- chosen artifact or reconstruction path

Pseudo-logic:

1. Compute exact key.
2. If exact artifact exists and passes admissibility, return L0.
3. Compute cheap fingerprint $F_{\text{cheap}}$.
4. Use graph neighborhood to retrieve a bounded L1 candidate set.
5. Rank admissible candidates by reuse risk + concept overlap + freshness.
6. If best candidate passes threshold, return L1.
7. Else reconstruct via L2.

Cái hay ở đây là:
- CAG gốc = step 1 + 2 + 7
- GraphCAG = thêm step 3,4,5,6

Vậy algorithm của bạn không phủ định CAG.
Nó literally extends CAG bằng một intermediate layer.

**Bài toán bạn thực sự giải**
Bạn cần nêu bài toán theo đúng kiểu research problem, không theo kiểu system integration.

Bài toán nên viết là:

Given a sequence of requests under evolving state, design a reuse controller that minimizes reconstruction cost while preserving correctness under drift.

Mục tiêu tối ưu:

$$
\min \; \mathbb{E}[\text{cost}]
$$

subject to:

$$
\Pr(\text{unsafe reuse}) \le \epsilon
$$

Đây là formulation rất đẹp.

Nó nói:
- mục tiêu là efficiency
- ràng buộc là correctness/safety

Và GraphCAG giải bài toán này bằng:
- admissibility theory: PCC
- controller algorithm: RAPID
- graph-local reusable unit: memory cell $M_g$

Ba cái này ghép lại thành xương sống của paper.

**Điểm ăn tiền nhất**
Nếu muốn paper mạnh mà chưa cần benchmark lớn, hãy xây theo chuỗi logic này:

1. Problem:
CAG exact reuse fails to characterize safe reuse under drift.
2. Theory:
PCC defines admissibility under drift.
3. Structure:
graph-local memory cell is the reusable unit.
4. Algorithm:
RAPID decides L0/L1/L2.
5. Reduction:
vanilla CAG is recovered as the L0-only special case.
6. Expressivity:
GraphCAG strictly extends CAG for drift-conditioned reuse.
7. Practice:
Stage 1 benchmark validates the executable lower bound.

Đây là cách làm paper mạnh thật sự.

**Bạn nên nói gì**
Nên nói:

- GraphCAG extends CAG from exact reuse to state-sensitive hierarchical reuse.
- Vanilla CAG is the L0 special case of GraphCAG.
- The main novelty is not another cache backend, but an admissibility theory and a hierarchical reuse controller under drift.
- GraphCAG solves a strictly richer decision problem than vanilla CAG.

Không nên nói:

- GraphCAG is not full CAG.
- We do not solve CAG yet.
- GraphCAG is better than CAG everywhere.
- We only have a lower bound so the contribution is limited.

**LaTeX-ready thesis**
Bạn có thể đưa gần như nguyên văn này vào paper:

```tex
\paragraph{Problem formulation.}
We study reuse under state drift. Given a current request-state tuple
$x = (q,p,s)$ and a set of previously materialized inference artifacts,
the system must choose among exact reuse, graph-aware near-hit reuse,
and full grounded reconstruction so as to minimize inference cost while
keeping unsafe reuse below a prescribed tolerance.

\paragraph{Core theoretical claim.}
Vanilla CAG is the exact-reuse special case of GraphCAG. When the
controller is restricted to exact-key lookup only, with no graph-local
candidate expansion and no admissible near-hit reuse, GraphCAG reduces
to a standard CAG fast path followed by reconstruction on miss.

\paragraph{Why GraphCAG is strictly richer.}
GraphCAG extends CAG to settings where exact reuse is insufficient
because user state, task intent, or progress may drift across requests.
Its additional expressive power comes from two elements: (i) PCC, which
defines when reuse remains valid under drift, and (ii) an intermediate
graph-aware reuse layer that searches for admissible near hits before
triggering full reconstruction.
```

Và cho algorithm:

```tex
\paragraph{Algorithmic contribution.}
The RAPID controller operationalizes this formulation by executing a
three-level policy: L0 exact reuse, L1 graph-aware near-hit reuse, and
L2 grounded reconstruction. In this sense, GraphCAG does not replace
CAG; it contains CAG as its lowest-cost exact-reuse layer and extends
it with a principled controller for drift-conditioned reuse.
```

**Kết luận thẳng**
Hướng đúng là:

- không phòng thủ
- không khoe benchmark quá sớm
- không phủ định CAG
- xây một theory of admissible reuse under drift
- xây một algorithm of hierarchical reuse control
- chứng minh CAG là special case của GraphCAG

Nếu làm như vậy, paper của bạn mạnh lên vì:
- có problem formulation rõ
- có theorem-level positioning rõ
- có algorithm-level contribution rõ
- benchmark chỉ là validation, không phải chỗ duy nhất để “xin điểm”

1. Tôi có thể sửa trực tiếp paper theo đúng trục này: thêm “Problem formulation”, “Vanilla CAG as a special case”, và viết lại đoạn CAG/Related Work/Conclusion.
2. Tôi cũng có thể giúp bạn viết hẳn 2 định lý và 1 proposition theo phong cách academic LaTeX để nhét vào section theory/algorithms.