from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from run_interventions import _load_dataset_records, _load_embedding_array
    

def visualize_mds_3d_for_quad(
    i: int,
    j: int,
    k: int,
    l: int,
    embeddings_simple: np.ndarray,
    embeddings_all_composite: np.ndarray,
    embeddings_simple_left: np.ndarray = None,
    embeddings_simple_right: np.ndarray = None,
    obj_means: np.ndarray = None,
    attr_means: np.ndarray = None,
    use_cosine: bool = True,
    use_half: bool = False,
    random_state: int = 0,
    save_path: Path = None,
    show: bool = True,
    annotate: bool = True,
    include_synthetic: bool = True,
    engine: str = "matplotlib",
    # Spherical projection options
    spherical: bool = False,
    sphere_radius: float = 1.0,
    sphere_opacity: float = 0.12,
):
    """
    3D MDS visualization for quadruple (i, j, k, l) with BCOS annotation scheme.

    Mapping of indices to letters:
      (i, j, k, l) -> (B, C, R, S)
        - B: attribute mean of i (first attribute)
        - C: object mean of j (first object)
        - R: attribute mean of k (second attribute)
        - S: object mean of l (second object)

    Points included (labels):
      - Means: B, C, R, S
      - Simple embeddings (E_simple): BC=S_ij, BS=S_il, RC=S_kj, RS=S_kl
      - Synthetic sums:
          B+C=A_i+O_j, B+S=A_i+O_l, R+C=A_k+O_j, R+S=A_k+O_l
          BC + OS = S_ij + S_kl, BS + OC = S_il + S_kj
      - Composite embeddings (E_comp):
          BCOS = C_{i,j,k,l}
          BSOC = C_{i,l,k,j}
    Returns a dict with labels, categories, 3D coordinates, distance matrix,
    and stress values.
    """
    import matplotlib.pyplot as plt
    try:
        from sklearn.manifold import MDS
    except ImportError as e:
        raise ImportError("scikit-learn required for MDS. pip install scikit-learn") from e

    # Optional: Plotly for interactive rendering
    plotly_available = False
    if engine.lower() == "plotly":
        try:
            import plotly.graph_objects as go  # type: ignore
            plotly_available = True
        except Exception:
            plotly_available = False

    # try to enable 3D plotting (modern Matplotlib doesn't strictly require this import)
    try:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except Exception:
        pass

    # Validate indices
    n = embeddings_simple.shape[0]
    for name, idx in zip(["i", "j", "k", "l"], [i, j, k, l]):
        if not (0 <= idx < n):
            raise ValueError(f"{name}={idx} out of range (n={n})")

    # Helper functions
    def l2norm(v: np.ndarray) -> np.ndarray:
        return v / (np.linalg.norm(v) + 1e-12)

    def add_if_nonzero(label: str, cat: str, v: np.ndarray, eps: float = 1e-9):
        """Add vector to visualization if it's non-zero and not duplicate."""
        if label in labels or v is None or float(np.linalg.norm(v)) <= eps:
            return
        vecs.append(l2norm(v.astype(np.float32)))
        labels.append(label)
        cats.append(cat)

    # Data structures for points
    vecs: List[np.ndarray] = []
    labels: List[str] = []
    cats: List[str] = []

    # IMPROVED COLOR SCHEME - More meaningful and clearer
    # Using a carefully designed palette with distinct purposes
    color_map = {
        # Basic building blocks - unified color for all concepts
        "attr": "#3498db",           # Blue - attributes
        "obj": "#3498db",            # Blue - objects (same as attributes)
        
        # Observed embeddings - green tones (what we actually measure)
        "simple": "#27ae60",         # Green - simple observed pairs
        
        # Predicted/composite embeddings - dark navy/indigo
        "composite": "#34495e",      # Dark blue-gray - main composite (BCRS, BSRC)
        
        # Synthetic predictions - purple/violet (theoretical combinations)
        "synthetic_basic": "#9b59b6", # Purple - basic synthetic (B+C, R+S, etc.)
        "synthetic_double": "#8e44ad", # Darker purple - double sums (BC+RS, etc.)
    }

    mu = np.mean(embeddings_simple, axis=(0, 1), keepdims=False)

    print(f"Mean norm of composite embeddings (mu): {np.linalg.norm(mu):.4f}")
    print(f"Mean norm of simple embeddings: {np.linalg.norm(embeddings_simple.mean(axis=(0,1))):.4f}")
    print(f"Mean norm of object means: {np.linalg.norm(obj_means.mean(axis=0)) if obj_means is not None else 'N/A'}")
    print(f"Mean norm of attribute means: {np.linalg.norm(attr_means.mean(axis=0)) if attr_means is not None else 'N/A'}")

    if attr_means is None and obj_means is None:
        embeddings_simple_centered = embeddings_simple - mu

        attr_means = np.mean(embeddings_simple_centered, axis=1, keepdims=False)
        obj_means = np.mean(embeddings_simple_centered, axis=0, keepdims=False)

        # Means according to BCOS mapping
        A_i = attr_means[i] + mu  # B
        O_j = obj_means[j] + mu  # C
        A_k = attr_means[k] + mu  # O
        O_l = obj_means[l] + mu  # S
    else:
        A_i = attr_means[i] # B
        O_j = obj_means[j]  # C
        A_k = attr_means[k]  # O
        O_l = obj_means[l]  # S

    # Add means
    add_if_nonzero("B", "attr", A_i)
    add_if_nonzero("C", "obj", O_j)
    if k != i:
        add_if_nonzero("R", "attr", A_k)
    if l != j:
        add_if_nonzero("S", "obj", O_l)

    # Get simple embeddings (handle left/right split if available)
    if embeddings_simple_left is not None and embeddings_simple_right is not None:
        S_ij = embeddings_simple_left[i, j] - mu
        S_il = embeddings_simple_right[i, l] - mu
        S_kj = embeddings_simple_left[k, j] - mu
        S_kl = embeddings_simple_right[k, l] - mu
    else:
        S_ij = embeddings_simple[i, j] - mu
        S_il = embeddings_simple[i, l] - mu
        S_kj = embeddings_simple[k, j] - mu
        S_kl = embeddings_simple[k, l] - mu

    # Add simple embeddings
    add_if_nonzero("BC", "simple", S_ij + mu)
    add_if_nonzero("BS", "simple", S_il + mu)
    add_if_nonzero("RC", "simple", S_kj + mu)
    add_if_nonzero("RS", "simple", S_kl + mu)

    # Add primary composite embeddings
    add_if_nonzero("BCRS", "composite", embeddings_all_composite[i, j, k, l])
    if (i, l, k, j) != (i, j, k, l):
        add_if_nonzero("BSRC", "composite", embeddings_all_composite[i, l, k, j])

    # Add additional composite embeddings (various permutations)
    composite_mappings = [
        ("BCRC", (i, j, k, j)),
        ("BSRS", (i, l, k, l)),
        ("BCBS", (i, j, i, l)),
        ("RCRS", (k, j, k, l)),
        # ("RSBC", (k, l, i, j)),  # Duplicate of BCRS, removed
        # ("RCBS", (k, j, i, l)),  # Duplicate of BSRC, removed
        # ("RCBC", (k, j, i, j)),  # Duplicate of BCRC, removed
        # ("RSBS", (k, l, i, l)),  # Duplicate of BSRS, removed
        # ("BSBC", (i, l, i, j)),  # Duplicate of BCBS, removed
        # ("RSRC", (k, l, k, j)),  # Duplicate of RCRS, removed
        ("RSRS", (k, l, k, l)),
        ("BSBS", (i, l, i, l)),
        ("BCBC", (i, j, i, j)),
        ("RCRC", (k, j, k, j)),
    ]
    for label, (idx_i, idx_j, idx_k, idx_l) in composite_mappings:
        add_if_nonzero(label, "composite", embeddings_all_composite[idx_i, idx_j, idx_k, idx_l])

    # Add synthetic sums
    if include_synthetic:
        # Single attribute + object sums (basic synthetic)
        add_if_nonzero("B+C", "synthetic_basic", A_i + O_j - mu)
        add_if_nonzero("B+S", "synthetic_basic", A_i + O_l - mu)
        add_if_nonzero("R+C", "synthetic_basic", A_k + O_j - mu)
        add_if_nonzero("R+S", "synthetic_basic", A_k + O_l - mu)

        # Double sums (with optional half-weighting)
        weight = 0.5 if use_half else 1.0
        double_sums = [
            ("BC + RS", S_ij + S_kl),
            ("RC + BS", S_il + S_kj),
            ("BC + RC", S_ij + S_kj),
            ("BS + RS", S_il + S_kl),
            ("BC + BS", S_ij + S_il),
            ("RC + RS", S_kj + S_kl),
        ]
        for label, sum_vec in double_sums:
            add_if_nonzero(label, "synthetic_double", weight * sum_vec + mu)


    # Compute distance matrix
    V = np.stack(vecs, 0)
    if use_cosine:
        Vn = V / (np.linalg.norm(V, axis=-1, keepdims=True) + 1e-12)
        sim = Vn @ Vn.T
        dist = 1.0 - sim
        np.fill_diagonal(dist, 0.0)
        dist = np.clip(dist, 0.0, 2.0)
    else:
        diff = V[:, None, :] - V[None, :, :]
        dist = np.linalg.norm(diff, axis=-1)

    # Perform 3D MDS
    mds = MDS(
        n_components=3,
        dissimilarity="precomputed",
        random_state=random_state,
        n_init=4,
        max_iter=300,
    )
    coords = mds.fit_transform(dist)

    raw_stress = float(getattr(mds, "stress_", np.nan))
    denom = float((dist ** 2).sum() + 1e-12)
    normalized_stress = raw_stress / denom

    # Perform PCA for 3D projection (faster and deterministic)
    # from sklearn.decomposition import PCA
    # pca = PCA(n_components=3, random_state=random_state)
    # coords = pca.fit_transform(V)
    # raw_stress = 1
    # normalized_stress = 1

    # Optional: project to a sphere
    if spherical:
        ctr = coords.mean(axis=0, keepdims=True)
        X = coords - ctr
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        X_unit = X / norms
        coords = X_unit * float(sphere_radius)

    # Build helper index map from labels to coordinates
    label_to_idx: Dict[str, int] = {lab: t for t, lab in enumerate(labels)}

    # IMPROVED LINE CONNECTIONS - organized by semantic meaning
    # Format: (label1, label2, color, linewidth, linestyle, legend_name, show_in_legend)
    
    # Define connection groups with clear purposes
    line_connections = [
        # GROUP 1: Concept to simple embedding connections (attr/obj → simple pairs)
        # These show how basic concepts combine into observed embeddings
        ("B+C", "BC", "#9b59b6", 3, "dash", "concept→simple (B+C→BC)", True),
        ("R+S", "RS", "#9b59b6", 3, "dash", "concept→simple (R+S→RS)", False),
        ("R+C", "RC", "#9b59b6", 3, "dash", "concept→simple (R+C→RC)", False),
        ("B+S", "BS", "#9b59b6", 3, "dash", "concept→simple (B+S→BS)", False),
        
        # GROUP 2: Cross-connections between simple embeddings
        # These show relationships between different observed pairs
        # ("RS", "BC", "#27ae60", 4, "solid", "cross-simple (RS↔BC)", True),
        # ("RC", "BS", "#27ae60", 4, "solid", "cross-simple (RC↔BS)", False),
        
        # GROUP 3: Synthetic cross-connections
        # Theoretical relationships between synthetic sums
        # ("R+C", "B+S", "#e67e22", 3, "dot", "cross-synthetic (R+C↔B+S)", True),
        # ("R+S", "B+C", "#e67e22", 3, "dot", "cross-synthetic (R+S↔B+C)", False),
        
        # GROUP 4: Double sum to composite connections
        # How sums of simple embeddings relate to composite embeddings
        ("BC + RS", "BCRS", "#34495e", 4, "solid", "sum→composite", True),
        # ("RC + BS", "RCBS", "#34495e", 4, "solid", None, False),  # RCBS is duplicate of BSRC
        # ("BC + RS", "RSBC", "#34495e", 4, "solid", None, False),  # RSBC is duplicate of BCRS
        ("RC + BS", "BSRC", "#34495e", 4, "solid", None, False),
        
        # GROUP 5: Special composite connections (partial overlaps)
        ("RC + RS", "RCRS", "#34495e", 3, "solid", "partial→composite", True),
        ("BC + BS", "BCBS", "#34495e", 3, "solid", None, False),
        ("BS + RS", "BSRS", "#34495e", 3, "solid", None, False),
        ("BC + RC", "BCRC", "#34495e", 3, "solid", None, False),
        # ("RC + RS", "RSRC", "#34495e", 3, "solid", None, False),  # RSRC is duplicate of RCRS
        # ("BC + BS", "BSBC", "#34495e", 3, "solid", None, False),  # BSBC is duplicate of BCBS
        # ("BS + RS", "RSBS", "#34495e", 3, "solid", None, False),  # RSBS is duplicate of BSRS
        # ("BC + RC", "RCBC", "#34495e", 3, "solid", None, False),  # RCBC is duplicate of BCRC
        
        # GROUP 6: Identity connections (double embeddings)
        ("RC", "RCRC", "#34495e", 3, "solid", "identity (X→XX)", True),
        ("BS", "BSBS", "#34495e", 3, "solid", None, False),
        ("BC", "BCBC", "#34495e", 3, "solid", None, False),
        ("RS", "RSRS", "#34495e", 3, "solid", None, False),
    ]

    # Define cycles with clear semantic meaning
    cycles = [
        # Synthetic concept cycle - shows how basic concepts relate
        (["B+C", "R+C", "R+S", "B+S"], "#e67e22", "synthetic cycle (concepts)"),
        # Simple embedding cycle - shows observed embedding relationships  
        (["BC", "RC", "RS", "BS"], "#27ae60", "simple cycle (observed)"),
    ]

    # Render with appropriate backend
    _render_visualization(
        coords=coords,
        labels=labels,
        cats=cats,
        label_to_idx=label_to_idx,
        color_map=color_map,
        line_connections=line_connections,
        cycles=cycles,
        title=f"3D MDS ({i},{j},{k},{l}) stress={raw_stress:.2e} norm={normalized_stress:.2e}",
        annotate=annotate,
        spherical=spherical,
        sphere_radius=sphere_radius,
        sphere_opacity=sphere_opacity,
        engine=engine,
        plotly_available=plotly_available,
        save_path=save_path,
        show=show,
    )

    return {
        "labels": np.array(labels),
        "category": np.array(cats),
        "coords": coords,
        "dist_matrix": dist,
        "raw_stress": np.array(raw_stress),
        "normalized_stress": np.array(normalized_stress),
    }

def _render_visualization(
    coords,
    labels,
    cats,
    label_to_idx,
    color_map,
    line_connections,
    cycles,
    title,
    annotate,
    spherical,
    sphere_radius,
    sphere_opacity,
    engine,
    plotly_available,
    save_path,
    show,
):
    """Unified rendering function for both Plotly and Matplotlib."""
    if plotly_available:
        import plotly.graph_objects as go
        fig = go.Figure()

        # Optional: add a translucent sphere background
        if spherical:
            phi = np.linspace(0, np.pi, 40)
            theta = np.linspace(0, 2 * np.pi, 80)
            xs = sphere_radius * np.outer(np.sin(phi), np.cos(theta))
            ys = sphere_radius * np.outer(np.sin(phi), np.sin(theta))
            zs = sphere_radius * np.outer(np.cos(phi), np.ones_like(theta))
            fig.add_trace(
                go.Surface(
                    x=xs, y=ys, z=zs,
                    opacity=float(sphere_opacity),
                    showscale=False,
                    hoverinfo="skip",
                    colorscale=[[0, "#DDDDDD"], [1, "#DDDDDD"]],
                    name=f"sphere r={sphere_radius:g}",
                    showlegend=False,
                )
            )

        # Scatter points per category with improved organization
        category_display_names = {
            "attr": "Concepts (attributes)",
            "obj": "Concepts (objects)",
            "simple": "Simple pairs (observed)",
            "composite": "Composite (predicted)",
            "synthetic_basic": "Synthetic basic (attr+obj)",
            "synthetic_double": "Synthetic double (pair+pair)",
        }
        
        marker_sizes = {
            "attr": 10,
            "obj": 10,
            "simple": 9,
            "composite": 11,
            "synthetic_basic": 8,
            "synthetic_double": 8,
        }
        
        for cat in ["attr", "obj", "simple", "composite", "synthetic_basic", "synthetic_double"]:
            idxs = [t for t, c in enumerate(cats) if c == cat]
            if not idxs:
                continue
            
            display_name = category_display_names.get(cat, cat)
            marker_size = marker_sizes.get(cat, 8)
            show_text_for_cat = annotate and cat not in {"synthetic_basic", "synthetic_double"}
            
            fig.add_trace(
                go.Scatter3d(
                    x=coords[idxs, 0],
                    y=coords[idxs, 1],
                    z=coords[idxs, 2],
                    mode="markers+text" if show_text_for_cat else "markers",
                    text=[labels[t] for t in idxs] if show_text_for_cat else None,
                    textposition="top center",
                    textfont=dict(size=20, color=color_map[cat]),
                    marker=dict(
                        size=marker_size,
                        color=color_map[cat],
                        line=dict(width=1, color='white')
                    ),
                    name=f"{display_name} ({len(idxs)})",
                    hovertext=[labels[t] for t in idxs],
                    hoverinfo="text",
                    legendgroup=cat,
                )
            )

        # Add cycles first (so they appear behind lines in legend)
        for cycle_labels, color, name in cycles:
            pts = [label_to_idx[l] for l in cycle_labels if l in label_to_idx]
            if len(pts) < 2:
                continue
            xs = list(coords[pts, 0]) + [coords[pts[0], 0]]
            ys = list(coords[pts, 1]) + [coords[pts[0], 1]]
            zs = list(coords[pts, 2]) + [coords[pts[0], 2]]
            fig.add_trace(
                go.Scatter3d(
                    x=xs, y=ys, z=zs,
                    mode="lines",
                    line=dict(color=color, width=12),
                    name=name,
                    showlegend=True,
                    legendgroup="cycles",
                )
            )

        # Add line connections organized by group
        linestyle_map = {"solid": "solid", "dash": "dash", "dot": "dot"}
        for label1, label2, color, width, style, name, show_legend in line_connections:
            if label1 in label_to_idx and label2 in label_to_idx:
                ia, ib = label_to_idx[label1], label_to_idx[label2]
                fig.add_trace(
                    go.Scatter3d(
                        x=[coords[ia, 0], coords[ib, 0]],
                        y=[coords[ia, 1], coords[ib, 1]],
                        z=[coords[ia, 2], coords[ib, 2]],
                        mode="lines",
                        line=dict(color=color, width=width*1.8, dash=linestyle_map[style]),
                        name=name,
                        showlegend=show_legend,
                        legendgroup="connections",
                    )
                )

        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
        spans = np.maximum(maxs - mins, 1e-6)
        pad = 0.18 * spans
        x_range = [float(mins[0] - pad[0]), float(maxs[0] + pad[0])]
        y_range = [float(mins[1] - pad[1]), float(maxs[1] + pad[1])]
        z_range = [float(mins[2] - pad[2]), float(maxs[2] + pad[2])]

        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor='center', font=dict(size=22)),
            width=1700,
            height=1200,
            scene=dict(
                xaxis=dict(
                    title="",
                    backgroundcolor="white",
                    showgrid=False,
                    zeroline=False,
                    showticklabels=False,
                    showaxeslabels=False,
                    range=x_range,
                ),
                yaxis=dict(
                    title="",
                    backgroundcolor="white",
                    showgrid=False,
                    zeroline=False,
                    showticklabels=False,
                    showaxeslabels=False,
                    range=y_range,
                ),
                zaxis=dict(
                    title="",
                    backgroundcolor="white",
                    showgrid=False,
                    zeroline=False,
                    showticklabels=False,
                    showaxeslabels=False,
                    range=z_range,
                ),
                bgcolor="white",
            ),
            paper_bgcolor="white",
            margin=dict(l=80, r=220, t=90, b=70),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.01,
                font=dict(size=14),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgb(200,200,200)",
                borderwidth=1
            )
        )

        # Save/show
        if save_path is not None:
            try:
                fig.write_html(str(save_path), include_plotlyjs="cdn")
                print(f"[MDS-3D/plotly] Saved interactive HTML to {save_path}")
            except Exception as e:
                print(f"[MDS-3D/plotly] Save failed: {e}")
        if show:
            try:
                fig.show()
            except Exception:
                pass

    else:
        # Matplotlib fallback
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Optional: draw a light wireframe sphere
        if spherical:
            u = np.linspace(0, 2 * np.pi, 40)
            v = np.linspace(0, np.pi, 20)
            xs = sphere_radius * np.outer(np.cos(u), np.sin(v))
            ys = sphere_radius * np.outer(np.sin(u), np.sin(v))
            zs = sphere_radius * np.outer(np.ones_like(u), np.cos(v))
            ax.plot_wireframe(xs, ys, zs, color="#CCCCCC", linewidth=0.5, alpha=0.5)

        # Scatter points
        category_display_names = {
            "attr": "Concepts (attributes)",
            "obj": "Concepts (objects)",
            "simple": "Simple pairs",
            "composite": "Composite",
            "synthetic_basic": "Synthetic basic",
            "synthetic_double": "Synthetic double",
        }
        
        marker_sizes = {
            "attr": 100,
            "obj": 100,
            "simple": 80,
            "composite": 120,
            "synthetic_basic": 70,
            "synthetic_double": 70,
        }
        
        for cat in ["attr", "obj", "simple", "composite", "synthetic_basic", "synthetic_double"]:
            idxs = [t for t, c in enumerate(cats) if c == cat]
            if not idxs:
                continue
            
            xs, ys, zs = coords[idxs, 0], coords[idxs, 1], coords[idxs, 2]
            display_name = category_display_names.get(cat, cat)
            marker_size = marker_sizes.get(cat, 70)
            
            ax.scatter(xs, ys, zs, c=[color_map[cat]] * len(idxs),
                      label=f"{display_name} ({len(idxs)})",
                      s=marker_size, depthshade=True, edgecolors='white', linewidth=0.5)

        # Add cycles
        for cycle_labels, color, name in cycles:
            pts = [label_to_idx[l] for l in cycle_labels if l in label_to_idx]
            if len(pts) < 2:
                continue
            xs = list(coords[pts, 0]) + [coords[pts[0], 0]]
            ys = list(coords[pts, 1]) + [coords[pts[0], 1]]
            zs = list(coords[pts, 2]) + [coords[pts[0], 2]]
            ax.plot(xs, ys, zs, color=color, linewidth=4, label=name)

        # Add line connections
        linestyle_map = {"solid": "-", "dash": "--", "dot": ":"}
        added_legend = set()
        for label1, label2, color, width, style, name, show_legend in line_connections:
            if label1 in label_to_idx and label2 in label_to_idx:
                ia, ib = label_to_idx[label1], label_to_idx[label2]
                legend_label = name if (show_legend and name not in added_legend) else None
                if legend_label:
                    added_legend.add(name)
                ax.plot([coords[ia, 0], coords[ib, 0]],
                       [coords[ia, 1], coords[ib, 1]],
                       [coords[ia, 2], coords[ib, 2]],
                       color=color, linewidth=width*1.5, linestyle=linestyle_map[style],
                       label=legend_label)

        if annotate:
            for idx, ((x, y, z), lab) in enumerate(zip(coords, labels)):
                if cats[idx] in {"synthetic_basic", "synthetic_double"}:
                    continue
                ax.text(x, y, z, lab, fontsize=10, weight='bold')

        ax.set_title(title, pad=20)
        ax.set_xlabel("MDS-1")
        ax.set_ylabel("MDS-2")
        ax.set_zlabel("MDS-3")
        ax.legend(frameon=True, fontsize=7, loc="best", fancybox=True, framealpha=0.8)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()

        if save_path is not None:
            try:
                fig.savefig(str(save_path), dpi=150, bbox_inches='tight')
                print(f"[MDS-3D] Saved to {save_path}")
            except Exception as e:
                print(f"[MDS-3D] Save failed: {e}")
            plt.close(fig)
        elif show:
            plt.show()
        else:
            plt.close(fig)


def _build_composite_from_loaded(loaded):
    records = loaded.records
    emb = loaded.embeddings
    if emb.ndim != 2:
        raise ValueError(f"Expected loaded embeddings [N,D], got {emb.shape}")

    n_attr = max((r.attr1 for r in records), default=-1)
    n_attr = max(n_attr, max((r.attr2 for r in records), default=-1)) + 1
    n_obj = max((r.obj1 for r in records), default=-1)
    n_obj = max(n_obj, max((r.obj2 for r in records), default=-1)) + 1
    if n_attr <= 0 or n_obj <= 0:
        raise RuntimeError("Failed to infer attr/object cardinalities from loaded records.")

    d = int(emb.shape[1])
    sums = np.zeros((n_attr, n_obj, n_attr, n_obj, d), dtype=np.float32)
    counts = np.zeros((n_attr, n_obj, n_attr, n_obj), dtype=np.int32)

    for rec, vec in zip(records, emb):
        sums[rec.attr1, rec.obj1, rec.attr2, rec.obj2] += vec.astype(np.float32)
        counts[rec.attr1, rec.obj1, rec.attr2, rec.obj2] += 1

    present_mask = counts > 0
    embeddings_composite = np.zeros_like(sums)
    embeddings_composite[present_mask] = sums[present_mask] / counts[present_mask][..., None]
    return embeddings_composite, present_mask


def main():
    import argparse

    parser = argparse.ArgumentParser(description="MDS 3D visualization.")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to dataset.pkl (or folder containing it).")
    parser.add_argument("--embedding-path", type=str, required=True, help="Path to scene embeddings file (.pkl/.pt/.npy).")
    parser.add_argument("--adapter", type=str, required=True, choices=["clevr", "pug_spare", "text"], help="Dataset adapter, matched to run_interventions.")
    parser.add_argument("--world-name", type=str, default=None, help="Optional world name filter for pug_spare.")
    parser.add_argument("--require-character-pos-null", action="store_true", help="If set, keep only rows with null/empty character_pos for pug_spare.")
    parser.add_argument("--sample_indices", type=str, default="0,0,1,1", help="Comma-separated indices i,j,k,l.")
    parser.add_argument("--use_half", action="store_true", help="Use half weights for synthetic sums.")
    parser.add_argument("--use_euclidean", action="store_true", help="Use Euclidean distance instead of cosine.")
    parser.add_argument("--engine", type=str, default="plotly", choices=["plotly", "matplotlib"], help="Visualization backend.")
    parser.add_argument("--spherical", action="store_true", help="Project MDS coordinates onto a sphere and render a sphere.")
    parser.add_argument("--sphere-radius", type=float, default=1.0, help="Sphere radius for projection and rendering.")
    parser.add_argument("--sphere-opacity", type=float, default=0.12, help="Opacity for the sphere surface (plotly only).")
    args = parser.parse_args()

    embedding_path = Path(args.embedding_path)
    dataset_path = Path(args.dataset_path)

    adapter = args.adapter

    embeddings_np = _load_embedding_array(embedding_path)
    loaded = _load_dataset_records(
        dataset_path=dataset_path,
        embeddings=embeddings_np,
        adapter=adapter,
        world_name=args.world_name,
        require_character_pos_null=bool(args.require_character_pos_null),
    )
    embeddings_composite, present_mask = _build_composite_from_loaded(loaded)
    A, O, _, _, d = embeddings_composite.shape

    # Exclude identical object pairs from composite statistics:
    # (attr1, obj1, attr2, obj2) where (attr1 == attr2 and obj1 == obj2).
    a1 = np.arange(A)[:, None, None, None]
    o1 = np.arange(O)[None, :, None, None]
    a2 = np.arange(A)[None, None, :, None]
    o2 = np.arange(O)[None, None, None, :]
    not_identical_mask = ~((a1 == a2) & (o1 == o2))
    valid_composite_mask = present_mask & not_identical_mask

    print(f"Composite embeddings shape: {embeddings_composite.shape}")
    print(f"Observed composite entries: {int(present_mask.sum())}/{present_mask.size}")
    if not valid_composite_mask.any():
        raise RuntimeError("No valid composite entries after masking; check dataset-path and concept index mapping.")
    mu_composite = embeddings_composite[valid_composite_mask].mean(axis=0, keepdims=True).reshape(1, 1, 1, 1, d)
    print(f"Composite mean vector norm: {np.linalg.norm(mu_composite):.4f}")
    embeddings_composite_centered = embeddings_composite - mu_composite
    
    # derive embeddings_simple from embeddings_composite via averaging
    embeddings_simple = np.zeros((A, O, d), dtype=np.float32)

    for i in range(A):
        for j in range(O):
            v1 = embeddings_composite_centered[i, j].reshape(-1, d)
            v2 = embeddings_composite_centered[:, :, i, j].reshape(-1, d)
            m1 = valid_composite_mask[i, j].reshape(-1)
            m2 = valid_composite_mask[:, :, i, j].reshape(-1)
            vecs = np.concatenate([v1, v2], axis=0)
            valid = np.concatenate([m1, m2], axis=0)
            mask = valid & (np.linalg.norm(vecs, axis=1) > 0)

            if mask.any():
                embeddings_simple[i, j] = vecs[mask].mean(axis=0)

    embeddings_simple += mu_composite.reshape(d)

    print(f"Norm of simple embeddings mean vector: {np.linalg.norm(embeddings_simple.mean(axis=(0,1))):.4f}")


    object_embeddings = np.zeros((O, d), dtype=np.float32)
    for i in range(O):
        v1 = embeddings_composite_centered[:, i, :, :].reshape(-1, d)      # obj1 == i
        v2 = embeddings_composite_centered[:, :, :, i].reshape(-1, d)      # obj2 == i
        m1 = valid_composite_mask[:, i, :, :].reshape(-1)
        m2 = valid_composite_mask[:, :, :, i].reshape(-1)
        vecs = np.concatenate([v1, v2], axis=0)
        valid = np.concatenate([m1, m2], axis=0)
        mask = valid & (np.linalg.norm(vecs, axis=1) > 0)
        if mask.any():
            object_embeddings[i, :] = vecs[mask].mean(axis=0)
    object_embeddings += mu_composite.reshape(d)

    attribute_embeddings = np.zeros((A, d), dtype=np.float32)
    for i in range(A):
        v1 = embeddings_composite_centered[i, :, :, :].reshape(-1, d)      # attr1 == i
        v2 = embeddings_composite_centered[:, :, i, :].reshape(-1, d)      # attr2 == i
        m1 = valid_composite_mask[i, :, :, :].reshape(-1)
        m2 = valid_composite_mask[:, :, i, :].reshape(-1)
        vecs = np.concatenate([v1, v2], axis=0)
        valid = np.concatenate([m1, m2], axis=0)
        mask = valid & (np.linalg.norm(vecs, axis=1) > 0)
        if mask.any():
            attribute_embeddings[i, :] = vecs[mask].mean(axis=0)
    attribute_embeddings += mu_composite.reshape(d)

    print("Embeddings shapes:")
    print(embeddings_simple.shape)
    print(embeddings_composite.shape)

    i, j, k, l = [int(x) for x in args.sample_indices.split(",")]
    print(f"[MDS-3D] Selected indices: i={i}, j={j}, k={k}, l={l}")
    if i > A - 1 or j > O - 1 or k > A - 1 or l > O - 1:
        raise ValueError(f"Indices out of range. Max attribute index: {A - 1}, Max object index: {O - 1}")

    out_ext = "html" if args.engine == "plotly" else "png"
    save_mds_3d_path = f"figures/mds_3d/mds_3d_{adapter}_{i}_{j}_{k}_{l}_{'05' if args.use_half else ''}{'_sphere' if args.spherical else ''}{'_euc' if args.use_euclidean else ''}.{out_ext}"

    _ = visualize_mds_3d_for_quad(
        i=i,
        j=j,
        k=k,
        l=l,
        embeddings_simple=embeddings_simple,
        embeddings_all_composite=embeddings_composite,
        embeddings_simple_left=None,
        embeddings_simple_right=None,
        obj_means=object_embeddings,
        attr_means=attribute_embeddings,
        use_cosine=not args.use_euclidean,
        use_half=args.use_half,
        show=False,
        annotate=True,
        include_synthetic=True,
        engine=args.engine,
        save_path=save_mds_3d_path,
        spherical=args.spherical,
        sphere_radius=args.sphere_radius,
        sphere_opacity=args.sphere_opacity,
    )

if __name__ == "__main__":
    main()