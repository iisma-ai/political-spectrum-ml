# Similar projects

Researched on 2026-11-02. Claims come from search summaries and abstracts, so check the original paper before relying on a specific number.

## Political science and network science

- **DW-NOMINATE:** the reference measure. [Schickler and co-authors](https://resolve.cambridge.org/core/journals/studies-in-american-political-development/article/substance-and-change-in-congressional-ideology-nominate-and-its-alternatives/681859D264663F5B92A3F6CD225D71ED) argue it handles rapid change poorly; [McCarty defends it](https://www.cambridge.org/core/product/0D5161CCD8C62BEC8DD5ECF8FA8DABEA).
- **[Nokken-Poole](https://legacy.voteview.com/Nokken-Poole.htm):** tracks member change from Congress to Congress against fixed cutting lines.
- **[Andris et al. 2015](https://keep.lib.asu.edu/system/files/c160/Rise_of_Partisanship_2015.pdf):** House voting networks over time.
- **[Waugh et al. 2009](https://ar5iv.labs.arxiv.org/html/0907.3509):** modularity as a polarization measure, without assuming party structure.
- **[Senate modularity study](https://arxiv.org/pdf/1006.5731):** modularity and DW-NOMINATE broadly agree on polarization levels.
- **[Moody and Mucha](https://arxiv.org/pdf/1705.02305):** Senate polarization, 1975 to 2012, found through a secondary source.
- **[Signed networks](https://arxiv.org/pdf/2303.16774):** polarization rose markedly in both chambers, 1973 to 2016.
- **[Clusterability study](https://ns20.cs.cornell.edu/abstracts/SIAMNS_2020_paper_16.pdf):** coalition-style structure may fit better than two groups.
- **[Supra-Laplacian clustering](https://arxiv.org/pdf/2409.11984):** clustering of time-evolving Senate networks.
- **[Zhang et al. 2008](https://ideas.repec.org/a/eee/phsmap/v387y2008i7p1705-1712.html):** cosponsorship networks.
- **[Gentzkow, Shapiro, Taddy](https://www.nber.org/papers/w22423):** speech partisanship rose sharply from the early 1990s, and the published version addresses finite-sample bias.

## Machine learning and deep learning

- **[Kraft, Jain, Rush 2016](https://aclanthology.org/D16-1221.pdf):** multidimensional legislator vectors, about 90.6% average accuracy.
- **[NIPEN, AAAI 2018](https://arxiv.org/pdf/1904.11727):** neural ideal points from bill text.
- **[Party Matters, ACL 2018](https://arxiv.org/pdf/1805.08182):** text alone doesn't carry across sessions; author metadata helps.
- **[Yang et al., IJCAI 2020](https://www.ijcai.org/proceedings/2020/198):** graph convolution plus recurrent networks.
- **[EMNLP 2020 embedding model](https://aclanthology.org/2020.emnlp-main.46.pdf):** names time-varying legislator embeddings as an improvement to make.
- **[Mou et al., ACL 2021](https://aclanthology.org/2021.acl-long.99):** heterogeneous graph with votes and Twitter, with explicit timing.
- **[PAR 2022](https://ar5iv.labs.arxiv.org/html/2210.08362):** adds social and expert context.
- **[Text-based ideal points](https://arxiv.org/pdf/2005.04232):** ideal points from text.
- **LLM simulation:** [Political Actor Agent](https://www.researchgate.net/publication/386977460_Political_Actor_Agent_Simulating_Legislative_System_for_Roll_Call_Votes_Prediction_with_Large_Language_Models) and [European Parliament personas](https://arxiv.org/pdf/2506.11798).

## Possible gap

The search found no dedicated 2023 to 2025 paper on time-aware RNN legislator embeddings. This is limited evidence, not proof that none exists, so run a dedicated literature search before treating it as original work.
