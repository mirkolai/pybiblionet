# PyBiblioNet

**PyBiblioNet** is a Python library for performing *network-based bibliometrics*, an analytical framework that leverages network science to study and quantify relationships and influence among scientific entities such as authors, and articles.

## Why Use Network Analysis in Bibliometrics?

Network analysis provides powerful tools to uncover key patterns and actors in the scholarly ecosystem. By computing **centrality metrics**, it is possible to go beyond traditional indicators like raw citation counts or the h-index. Network-based metrics help identify:

- Influential authors or papers within and across research domains.
- Bridge entities that connect otherwise separated communities.
- Nodes that are highly connected to authoritative or prestigious sources.

These nuanced insights allow for a more refined understanding of scientific impact and visibility, as demonstrated in the literature (e.g., Diallo et al., 2016).

## Features

- Easy modeling of citation and collaboration networks.
- Calculation of a wide variety of centrality and influence metrics.
- Integration with common network science libraries (e.g., `networkx`, `igraph`).
- Support for directed and weighted graphs.
- Ready-to-use functions for common bibliometric scenarios.

## Installation

To install the latest version from TestPyPI:

```bash
pip install --upgrade pybiblionet --index-url https://test.pypi.org/simple/ pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple

