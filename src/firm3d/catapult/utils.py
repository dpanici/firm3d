__all__ = []


import numpy as np

__all__ = ["boozer_interpolant", "cartesian_interpolant"]


def boozer_interpolant(field, nfp, ns, ntheta, nzeta, vacuum=False):
    r"""
    Set up a Boozer vacuum interpolant for tracing.

    Args:
        field: BoozerMagneticField object
        nfp: Integer, number of field periods in the device
        n_meta_grid_pts: Integer, number of cells in s, theta, zeta
            to use for interpolation

    Returns:
        srange : (s_start, s_end, number of grid points in s for interpolation)
        trange : same as srange, but for theta
        zrange : same as srange, but for zeta
        cell_quad_pts : The interpolant data. Each row is a point in the grid,
            data is stored in columns modB, dmodBds, dmodBdtheta, dmodBdzeta, G, iota
        maximum observed J at grid points, useful for rejection sampling
    """
    srange = (0, 1.0, 3 * ns + 1)
    trange = (0, np.pi, 3 * ntheta + 1)
    zrange = (0, 2 * np.pi / nfp, 3 * nzeta + 1)

    s_grid = np.linspace(srange[0], srange[1], srange[2])
    theta_grid = np.linspace(trange[0], trange[1], trange[2])
    zeta_grid = np.linspace(zrange[0], zrange[1], zrange[2])

    quad_pts = np.empty((srange[2] * trange[2] * zrange[2], 3))
    for i in range(srange[2]):
        for j in range(trange[2]):
            for k in range(zrange[2]):
                quad_pts[trange[2] * zrange[2] * i + zrange[2] * j + k, :] = [
                    s_grid[i],
                    theta_grid[j],
                    zeta_grid[k],
                ]

    field.set_points(quad_pts)

    # Quantities to interpolate
    G = field.G()
    I = field.I()
    iota = field.iota()
    modB = field.modB()
    modB_derivs = field.modB_derivs()

    if vacuum:
        # Vacuum approximation: G=const, I=0, K=0
        quad_info = np.hstack((modB, modB_derivs, G, iota))
    else:
        # Full guiding center equations: include I and K
        dGds = field.dGds()
        dIds = field.dIds()
        K = field.K()
        K_derivs = field.K_derivs()
        quad_info = np.hstack((modB, modB_derivs, G, dGds, I, dIds, iota, K, K_derivs))

    # calculate max J for sampling
    J = (G + iota * I) / (modB**2)

    # reorder for device memory acceesses
    # print("reordering interpolant data from device accesses")
    s_ncells = int((srange[2] - 1) / 3)
    t_ncells = int((trange[2] - 1) / 3)
    z_ncells = int((zrange[2] - 1) / 3)
    cell_quad_pts = np.empty((s_ncells * t_ncells * z_ncells * 64, quad_info.shape[1]))

    for cell_s in range(s_ncells):
        for cell_t in range(t_ncells):
            for cell_z in range(z_ncells):
                row_start = 64 * (
                    cell_s * t_ncells * z_ncells + cell_t * z_ncells + cell_z
                )

                # iterate over spline locations for this cell
                for i in range(4):
                    for j in range(4):
                        for k in range(4):
                            row_idx = row_start + 16 * i + 4 * j + k
                            cell_quad_pts[row_idx, :] = quad_info[
                                trange[2] * zrange[2] * (3 * cell_s + i)
                                + zrange[2] * (3 * cell_t + j)
                                + 3 * cell_z
                                + k,
                                :,
                            ]
    cell_quad_pts = np.ascontiguousarray(cell_quad_pts)
    return srange, trange, zrange, cell_quad_pts, np.max(J)


def boozer_saw_interpolant(field, nfp, ns, ntheta, nzeta):
    r"""
    Set up a Boozer vacuum interpolant for tracing.

    Args:
        field: BoozerMagneticField object
        nfp: Integer, number of field periods in the device
        n_meta_grid_pts: Integer, number of cells in s, theta, zeta
            to use for interpolation

    Returns:
        srange : (s_start, s_end, number of grid points in s for interpolation)
        trange : same as srange, but for theta
        zrange : same as srange, but for zeta
        cell_quad_pts : The interpolant data. Each row is a point in the grid,
            data is stored in columns
            modB, dmodBds, dmodBdtheta, dmodBdzeta, G, dGds, I, dIds, iota, diotads
        maximum observed J at grid points, useful for rejection sampling
    """

    srange = (0, 1.0, 3 * ns + 1)
    trange = (0, np.pi, 3 * ntheta + 1)
    zrange = (0, 2 * np.pi / nfp, 3 * nzeta + 1)

    s_grid = np.linspace(srange[0], srange[1], srange[2])
    theta_grid = np.linspace(trange[0], trange[1], trange[2])
    zeta_grid = np.linspace(zrange[0], zrange[1], zrange[2])

    quad_pts = np.empty((srange[2] * trange[2] * zrange[2], 3))
    for i in range(srange[2]):
        for j in range(trange[2]):
            for k in range(zrange[2]):
                quad_pts[trange[2] * zrange[2] * i + zrange[2] * j + k, :] = [
                    s_grid[i],
                    theta_grid[j],
                    zeta_grid[k],
                ]

    field.set_points(quad_pts)

    # Quantities to interpolate
    G = field.G()
    iota = field.iota()
    modB = field.modB()
    modB_derivs = field.modB_derivs()
    dGds = field.dGds()
    I = field.I()
    dIds = field.dIds()
    diotads = field.diotads()
    quad_info = np.hstack((modB, modB_derivs, G, dGds, I, dIds, iota, diotads))

    # calculate max J for sampling
    I = field.I()
    J = (G + iota * I) / (modB**2)

    # reorder for device memory acceesses
    # print("reordering interpolant data from device accesses")
    s_ncells = int((srange[2] - 1) / 3)
    t_ncells = int((trange[2] - 1) / 3)
    z_ncells = int((zrange[2] - 1) / 3)
    cell_quad_pts = np.empty((s_ncells * t_ncells * z_ncells * 64, quad_info.shape[1]))

    for cell_s in range(s_ncells):
        for cell_t in range(t_ncells):
            for cell_z in range(z_ncells):
                row_start = 64 * (
                    cell_s * t_ncells * z_ncells + cell_t * z_ncells + cell_z
                )

                # iterate over spline locations for this cell
                for i in range(4):
                    for j in range(4):
                        for k in range(4):
                            row_idx = row_start + 16 * i + 4 * j + k
                            cell_quad_pts[row_idx, :] = quad_info[
                                trange[2] * zrange[2] * (3 * cell_s + i)
                                + zrange[2] * (3 * cell_t + j)
                                + 3 * cell_z
                                + k,
                                :,
                            ]
    cell_quad_pts = np.ascontiguousarray(cell_quad_pts)
    return srange, trange, zrange, cell_quad_pts, np.max(J)


def cartesian_interpolant(
    field,
    surface_classifier,
    rrange=None,
    zrange=None,
    phirange=None,
    source_grid=None,
    **kwargs,
):
    r"""
    Set up a Cartesian (cylindrical) interpolant for GPU tracing.

    Args:
        field: Magnetic field object. Supports either:

            - SIMSOPT ``InterpolatedField`` – grid ranges are read from the
              field's ``r_range``, ``phi_range``, ``z_range`` attributes.
            - DESC ``_MagneticField`` subclass – grid ranges must be supplied
              via *rrange*, *zrange*, *phirange*.
        surface_classifier: SurfaceClassifier object used to evaluate the
            signed distance to the plasma boundary at each grid point.
        rrange: tuple ``(r_start, r_end, n_cells)`` for the R grid.
            Required for DESC fields, ignored for SIMSOPT fields.
        zrange: tuple ``(z_start, z_end, n_cells)`` for the Z grid.
            Required for DESC fields, ignored for SIMSOPT fields.
        phirange: tuple ``(phi_start, phi_end, n_cells)`` for the phi grid.
            Required for DESC fields, ignored for SIMSOPT fields.
        source_grid: Grid used to discretize a DESC ``_MagneticField`` when
            computing B via Biot-Savart. Ignored for SIMSOPT fields.
            Default is ``None``.
        **kwargs: Additional keyword arguments. For DESC fields,
            ``chunk_size`` (default 100) controls the chunk size used by
            ``vmap_chunked`` when computing the gradient of |B|.

    Returns:
        r_range : (r_start, r_end, number of grid points in r)
        phi_range : (phi_start, phi_end, number of grid points in phi)
        z_range : (z_start, z_end, number of grid points in z)
        cell_quad_pts : Interpolant data reordered for GPU access. Shape is
            ``(n_cells * 64, n_features)`` where columns contain the magnetic
            field, gradient of |B|, and signed distance function values at
            the spline quadrature nodes for each cell.
    """
    # ---- Detect field type ------------------------------------------------
    try:
        from simsopt.field import InterpolatedField

        is_simsopt = isinstance(field, InterpolatedField)
    except ImportError:
        is_simsopt = False

    if is_simsopt:
        # SIMSOPT: ranges come from the field object
        r_range = (field.r_range[0], field.r_range[1], 3 * field.r_range[2] + 1)
        phi_range = (field.phi_range[0], field.phi_range[1], 3 * field.phi_range[2] + 1)
        z_range = (field.z_range[0], field.z_range[1], 3 * field.z_range[2] + 1)
        n_r_cells = field.r_range[2]
        n_phi_cells = field.phi_range[2]
        n_z_cells = field.z_range[2]
    else:
        # DESC (or other): ranges must be provided by the user
        if rrange is None or zrange is None or phirange is None:
            raise ValueError(
                "rrange, zrange, and phirange must be provided for "
                "non-SIMSOPT fields. Each should be a tuple "
                "(start, end, n_cells)."
            )
        r_range = (rrange[0], rrange[1], 3 * rrange[2] + 1)
        phi_range = (phirange[0], phirange[1], 3 * phirange[2] + 1)
        z_range = (zrange[0], zrange[1], 3 * zrange[2] + 1)
        n_r_cells = rrange[2]
        n_phi_cells = phirange[2]
        n_z_cells = zrange[2]

    # ---- Build cylindrical grid -------------------------------------------
    r_grid = np.linspace(r_range[0], r_range[1], r_range[2])
    phi_grid = np.linspace(phi_range[0], phi_range[1], phi_range[2])
    z_grid = np.linspace(z_range[0], z_range[1], z_range[2])

    quad_pts = np.empty((r_range[2] * phi_range[2] * z_range[2], 3))
    for i in range(r_range[2]):
        for j in range(phi_range[2]):
            for k in range(z_range[2]):
                quad_pts[phi_range[2] * z_range[2] * i + z_range[2] * j + k, :] = [
                    r_grid[i],
                    phi_grid[j],
                    z_grid[k],
                ]

    # ---- Compute B and GradAbsB ------------------------------------------
    if is_simsopt:
        field.set_points_cyl(quad_pts)
        B = field.B_cyl()
        GradAbsB = field.GradAbsB_cyl()
    else:
        # DESC path
        import jax
        import jax.numpy as jnp
        from desc.batching import vmap_chunked
        from desc.backend import print_backend_info

        print_backend_info()
        print("computing DESC field")
        B = np.array(
            field.compute_magnetic_field(quad_pts, basis="rpz", source_grid=source_grid)
        )
        print("finished computing DESC field")

        # Compute GradAbsB via JAX autodiff + chunked vmap
        def absB_at_point(rpz):
            """Compute |B| at a single (R, phi, Z) point."""
            B_rpz = field.compute_magnetic_field(
                rpz.reshape(1, 3), basis="rpz", source_grid=source_grid
            )
            return jnp.sqrt(jnp.sum(B_rpz**2))

        grad_absB_fn = jax.grad(absB_at_point)
        chunk_size = kwargs.get("chunk_size", 1000)
        print("computing desc field gradient")
        GradAbsB = np.array(
            vmap_chunked(grad_absB_fn, chunk_size=chunk_size)(jnp.array(quad_pts))
        )
        print("finished computing desc field gradient")

    # ---- Signed distance + combine features -------------------------------
    signed_dist_vals = surface_classifier.evaluate_rphiz(quad_pts)

    quad_info = np.hstack((B, GradAbsB, signed_dist_vals))

    # ---- Reorder for device memory accesses -------------------------------
    cell_quad_pts = np.empty(
        (
            n_r_cells * n_z_cells * n_phi_cells * 64,
            quad_info.shape[1],
        )
    )
    for cell_r in range(n_r_cells):
        for cell_phi in range(n_phi_cells):
            for cell_z in range(n_z_cells):
                row_start = 64 * (
                    cell_r * n_phi_cells * n_z_cells + cell_phi * n_z_cells + cell_z
                )

                assert 3 * cell_r + i < r_range[2]
                # iterate over spline locations for this cell
                for i in range(4):
                    for j in range(4):
                        for k in range(4):
                            row_idx = row_start + 16 * i + 4 * j + k

                            cell_quad_pts[row_idx, :] = quad_info[
                                phi_range[2] * z_range[2] * (3 * cell_r + i)
                                + z_range[2] * (3 * cell_phi + j)
                                + 3 * cell_z
                                + k,
                                :,
                            ]

    cell_quad_pts = np.ascontiguousarray(cell_quad_pts)

    return r_range, phi_range, z_range, cell_quad_pts
