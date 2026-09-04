# OpenVaccine model

`OpenVaccineGCNGRU` jointly consumes nucleotide identity, dot-bracket secondary structure and loop type. Base-pair edges undergo graph propagation before a two-layer bidirectional GRU and regression head. It is trained from scratch and follows the GCN+GRU family evaluated by iVaccine-Deep rather than a pretrained RNA model. Reference: Zrimec et al., PeerJ Computer Science 2021, doi:10.7717/peerj-cs.597; Wayment-Steele et al., Nature Machine Intelligence 2022.

