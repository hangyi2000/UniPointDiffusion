import meshplot as mp
import numpy as np
import fast_simplification


def export_html(gt_mesh, recon_mesh, surface, file_name,
                gt_simplify_ratio=0.9,
                recon_simplify_ratio=0.9,
                gap=1.1):

    if gt_simplify_ratio > 0:
        ratio = 1 - 1 / gt_simplify_ratio
        n_recon_f = int(np.array(gt_mesh.faces).shape[0] / gt_simplify_ratio)
        print(f"n_recon_f: {n_recon_f}, ratio = {ratio}")

        gtv, gtf = fast_simplification.simplify(
            gt_mesh.vertices.astype(np.float32), gt_mesh.faces, gt_simplify_ratio)
    else:
        gtv, gtf = gt_mesh.vertices, gt_mesh.faces

    if recon_simplify_ratio > 0:
        ratio = 1 - 1 / recon_simplify_ratio
        n_recon_f = int(np.array(gt_mesh.faces).shape[0] / recon_simplify_ratio)
        print(f"n_recon_f: {n_recon_f}, ratio = {ratio}")

        rev, ref = fast_simplification.simplify(
            recon_mesh.vertices.astype(np.float32), recon_mesh.faces, recon_simplify_ratio)
    else:
        rev, ref = recon_mesh.vertices, recon_mesh.faces

    # mp.offline()
    mp.website()
    # tmp = mp.plot(rev, ref, c=np.random.rand(*ref.shape))
    # tmp = mp.plot(rev, ref, c=np.array(rev[:, 1], dtype=np.float64))
    tmp = mp.plot(rev, ref)
    gtvx = gtv[:, 0] + gap
    # tmpx = gtv
    gtv[:, 0] = gtvx
    tmp.add_mesh(gtv, gtf)

    surface[:, 0] = surface[:, 0] - gap
    tmp.add_points(surface, shading={"point_size": 0.2})

    tmp.save(f"{file_name}")

    return 0

