import numpy as np
import cv2
import math
import hnswlib
import faiss
from sklearn.preprocessing import StandardScaler
from skimage.color import deltaE_ciede2000, rgb2lab

class GeneralContextMeasure:
    def __init__(self, beta2: float = 1.0, alpha: float = 5.0):
        """
        Initializes the ContextMeasure instance.

        Parameters:
            beta2 (float): Balancing factor for transmission and completeness.
            alpha (float): Scaling factor for Gaussian covariance.
            gamma (int): 
            lambda_spatial (float): 
        """
        self.beta2 = beta2
        self.alpha = alpha
        self._exp_factor = math.e / (math.e - 1)

    def compute(self, fm: np.ndarray, gt: np.ndarray, img: np.ndarray = None) -> float:
        """
        Computes the context measure between foreground map and ground truth.

        Parameters:
            fm (numpy.ndarray): foreground map (values between 0 and 255).
            gt (numpy.ndarray): ground truth map (values between 0 and 255).

        Returns:
            float: context measure value.
        """
        X = self._preprocess_map(fm, binary_flag = False)
        Y = self._preprocess_map(gt, binary_flag = True)
        cov_matrix, x_dis, y_dis = self._compute_y_params(Y)
        K = self._gaussian_kernel(x_dis, y_dis, cov_matrix)

        # CIR
        cir = self._calculate_cir(X, Y, K)
        mcir = np.sum(cir * X) / (np.sum(X) + 1e-8)

        # CAC
        cac = self._calculate_cac(X, Y, K)
        if img is not None:
            e = self._calculate_extrinsic(img, Y)
        else:
            e = np.zeros_like(Y)
        wcac = np.sum(cac * (Y + e)) / (np.sum(Y) + np.sum(e) + 1e-8)

        return (1 + self.beta2) * mcir * wcac / (self.beta2 * mcir + wcac + 1e-8)

    def _calculate_cir(self, X: np.ndarray, Y: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        x_binary = (X > 0).astype(int)
        global_relevance_matrix = cv2.filter2D(Y, cv2.CV_32F, kernel)
        return x_binary * global_relevance_matrix

    def _calculate_cac(self, X: np.ndarray, Y: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        X = X.astype(float)
        non_global_completeness_matrix = np.exp(-1 * cv2.filter2D(X, -1, kernel))
        global_completeness_matrix = 1 - non_global_completeness_matrix
        cac = self._exp_factor * Y * global_completeness_matrix
        return cac

    def _calculate_extrinsic(self, img: np.ndarray, Y: np.ndarray) -> np.ndarray:
        pass

    def _preprocess_map(self, img, binary_flag) -> np.ndarray:
        """
        Preprocesses the input map: converts to grayscale (normalizes fm, or binarizes gt).

        Parameters:
            img (numpy.ndarray): grayscale 0~255.
            binary_flag: True or False

        Returns:
            np.ndarray: (normalized grayscale fm, binarized gt)
        """
        # check img if it's a 2-channel image
        if img.ndim == 2:
            # Convert fm to grayscale
            if not binary_flag:
                img = img.astype(np.float64) / 255.0
            else:
            # Convert gt to grayscale 
                img = (img >= 128).astype(np.float64)  # binarize to 0 or 1

            return img
        else:
            print('img wrong')
            return None

    def _gaussian_kernel(self, x_dis: int, y_dis: int, cov_matrix: np.ndarray) -> np.ndarray:
        det_sigma = np.linalg.det(cov_matrix)
        inv_sigma = np.linalg.inv(cov_matrix)

        x, y = np.meshgrid(np.arange(-x_dis, x_dis + 1),
                           np.arange(-y_dis, y_dis + 1), indexing='ij')
        Z = np.stack([x, y], axis=-1)
        exp_term = np.einsum('...i,ij,...j->...', Z, inv_sigma, Z)

        kernel = np.exp(-0.5 * exp_term) / (2 * np.pi * np.sqrt(det_sigma))
        return kernel / np.sum(kernel)

    def _compute_y_params(self, Y: np.ndarray) -> tuple:
        points = np.argwhere(Y > 0)
        if len(points) <= 1:
            return np.diag([0.25, 0.25]), 1, 1

        cov_matrix = np.cov(points, rowvar=False)
        sigma_x = np.sqrt(cov_matrix[0, 0])
        sigma_y = np.sqrt(cov_matrix[1, 1])
        total_sigma = np.sqrt(cov_matrix[0, 0] + cov_matrix[1, 1])

        std_cov_matrix = self.alpha ** 2 * cov_matrix / (total_sigma ** 2)
        std_sigma_x = self.alpha * sigma_x / total_sigma
        std_sigma_y = self.alpha * sigma_y / total_sigma
        x_dis = round(3 * std_sigma_x)
        y_dis = round(3 * std_sigma_y)

        return std_cov_matrix, x_dis, y_dis

class CamoContextMeasure(GeneralContextMeasure):
    def __init__(self, beta2: float = 1.2, alpha: float = 5.0, gamma: int = 8, lambda_spatial: float = 20):
        """
        Initializes the ContextMeasure instance.

        Parameters:
            beta2 (float): Balancing factor for transmission and completeness.
            alpha (float): Scaling factor for Gaussian covariance.
            gamma (int): 
            lambda_spatial (float): 
        """
        self.beta2 = beta2
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_spatial = lambda_spatial
        self._exp_factor = math.e / (math.e - 1)

    def compute(self, fm: np.ndarray, gt: np.ndarray, img: np.ndarray = None) -> float:
        """
        Computes the context measure between foreground map and ground truth.

        Parameters:
            fm (numpy.ndarray): foreground map (values between 0 and 255).
            gt (numpy.ndarray): ground truth map (values between 0 and 255).

        Returns:
            float: context measure value.
        """
        X = self._preprocess_map(fm, binary_flag = False)
        Y = self._preprocess_map(gt, binary_flag = True)
        cov_matrix, x_dis, y_dis = self._compute_y_params(Y)
        K = self._gaussian_kernel(x_dis, y_dis, cov_matrix)

        # CIR
        cir = self._calculate_cir(X, Y, K)
        mcir = np.sum(cir * X) / (np.sum(X) + 1e-8)

        # CAC
        cac = self._calculate_cac(X, Y, K)
        if img is not None:
            _, cd = self._calculate_camouflage_degree(img, Y)
        else:
            cd = np.zeros_like(Y)
        wcac = np.sum(cac * (Y + cd)) / (np.sum(Y) + np.sum(cd) + 1e-8)

        return (1 + self.beta2) * mcir * wcac / (self.beta2 * mcir + wcac + 1e-8)

    def visualize(self, img: np.ndarray, gt: np.ndarray) -> tuple:
        Y = self._preprocess_map(gt, binary_flag = True)
        cov_matrix, x_dis, y_dis = self._compute_y_params(Y)
        kernel = self._gaussian_kernel(x_dis, y_dis, cov_matrix)
        img_recon, cd = self._calculate_camouflage_degree(img, Y)
        return img_recon, cd

    def _calculate_camouflage_degree(self, img: np.ndarray, mask: np.ndarray, w: int = 7) -> tuple:
        """
        Computes the camouflage degree matrix with Lab+spatial ANN and RGB reconstruction.

        Parameters:
        - img: BGR image (H x W x 3)
        - mask: binary mask (H x W)
        - w: patch size

        Returns:
        - img_recon: RGB reconstructed image
        - camouflage_degree_matrix: camouflage map (H x W)
        """

        mask_binary = (mask > 0).astype(np.uint8)
        fg_mask = mask_binary
        bg_mask = self._extract_surrounding_background(fg_mask, kernel_size=20)
        im_fg = fg_mask[:, :, np.newaxis] * img
        im_bg = bg_mask[:, :, np.newaxis] * img
        im_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        # Step 1: from Lab space to extract patch
        im_fg_lab = im_lab * fg_mask[:, :, np.newaxis]
        im_bg_lab = im_lab * bg_mask[:, :, np.newaxis]

        fg_indices, fg_feat_lab = self._extract_patches(im_fg_lab, fg_mask, w, d=w // 2)
        bg_indices, bg_feat_lab = self._extract_patches(im_bg_lab, bg_mask, w, d=w // 2)

        # Step 2: Lab+spatial for ANN query
        fg_nn = self._ann_with_spatial_faiss(
            bg_feat_lab, fg_feat_lab,
            bg_indices, fg_indices
        )

        # Step 3: reconstruct foreground in RGB space
        img_recon = self._reconstruct_image(
            img, fg_indices, bg_indices, fg_nn, im_bg, w
        )

        # Step 4: compute similarity in Lab space
        similarity_matrix = self._compute_delta_e2000_matrix(
            img_recon, im_fg.astype(np.uint8)
        ).astype(np.float64)

        # Step 5: compute Camouflage Degree
        cd = (
            (np.exp(self.gamma * similarity_matrix * mask_binary) - 1) / (np.exp(self.gamma) - 1)
        ).astype(np.float64)

        return img_recon, cd

    def _ann_with_spatial(self, x: np.ndarray, q: np.ndarray, x_coords: np.ndarray, q_coords: np.ndarray, m: int = 16) -> np.ndarray:
        """
        HNSWLib-based ANN search with spatial-aware feature augmentation.
        """
        # 标准化坐标
        scaler = StandardScaler()
        all_coords = np.vstack([x_coords, q_coords])
        scaled_coords = scaler.fit_transform(all_coords)
        x_coords_scaled = scaled_coords[:len(x_coords)]
        q_coords_scaled = scaled_coords[len(x_coords):]

        # 拼接图像特征 + 空间特征
        x_aug = np.hstack([x, self.lambda_spatial * x_coords_scaled])
        q_aug = np.hstack([q, self.lambda_spatial * q_coords_scaled])

        x_aug = x_aug.astype(np.float32)
        q_aug = q_aug.astype(np.float32)

        dim = x_aug.shape[1]
        num_elements = x_aug.shape[0]

        # 初始化 HNSW Index
        index = hnswlib.Index(space='l2', dim=dim)
        index.init_index(max_elements=num_elements, ef_construction=200, M=m, random_seed=42)
        index.add_items(x_aug)
        index.set_ef(64)

        # 查询最近邻
        k = 1
        indices, _ = index.knn_query(q_aug, k)
        return indices

    def _ann_with_spatial_faiss(self, x, q, x_coords, q_coords, m=16):
        # 坐标标准化 + 空间增强
        scaler = StandardScaler()
        all_coords = np.vstack([x_coords, q_coords])
        scaled_coords = scaler.fit_transform(all_coords)
        x_coords_scaled = scaled_coords[:len(x_coords)]
        q_coords_scaled = scaled_coords[len(x_coords):]

        x_aug = np.hstack([x, self.lambda_spatial * x_coords_scaled]).astype(np.float32)
        q_aug = np.hstack([q, self.lambda_spatial * q_coords_scaled]).astype(np.float32)

        # 建立 Index
        dim = x_aug.shape[1]
        index = faiss.IndexFlatL2(dim)  # L2 距离
        index.add(x_aug)

        _, indices = index.search(q_aug, 1)  # top-1
        return indices

    def _extract_surrounding_background(self, mask: np.ndarray, kernel_size: int = 50) -> np.ndarray:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        surrounding_bg_mask = dilated_mask - mask
        return surrounding_bg_mask

    def _extract_patches(self, img: np.ndarray, mask: np.ndarray, w: int, d: int) -> tuple:
        h, w_, c = img.shape
        pad_h = (d - (h - w) % d) % d
        pad_w = (d - (w_ - w) % d) % d
        img_padded = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        mask_padded = np.pad(mask, ((0, pad_h), (0, pad_w)), mode='constant')

        new_h, new_w = img_padded.shape[:2]

        img_patches = np.lib.stride_tricks.sliding_window_view(
            img_padded, (w, w, img.shape[2])
        )[::d, ::d, 0, :, :, :]
        mask_patches = np.lib.stride_tricks.sliding_window_view(
            mask_padded, (w, w)
        )[::d, ::d, :, :]

        img_patches = img_patches.reshape(-1, w * w * c)
        mask_patches = mask_patches.reshape(-1, w, w)

        grid_x, grid_y = np.meshgrid(
            np.arange(0, new_h - w + 1, d),
            np.arange(0, new_w - w + 1, d),
            indexing='ij'
        )
        all_indices = np.column_stack((grid_x.ravel(), grid_y.ravel()))
        valid_idx = np.all(mask_patches > 0, axis=(1, 2))
        valid_indices = all_indices[valid_idx]
        valid_patches = img_patches[valid_idx]

        return valid_indices, valid_patches

    def _reconstruct_image(self, img: np.ndarray, fg_indices: np.ndarray, bg_indices: np.ndarray, fg_nn: np.ndarray, im_bg: np.ndarray, w: int) -> np.ndarray:
        img_recon = np.zeros_like(img, dtype=np.int64)
        counts = np.zeros(img.shape[:2]) + 1e-8

        fg_x, fg_y = fg_indices[:, 0], fg_indices[:, 1]
        nn_i_j = fg_nn[:, 0]
        cii, cjj = bg_indices[nn_i_j, 0], bg_indices[nn_i_j, 1]

        fg_x = np.clip(fg_x, 0, img.shape[0] - w)
        fg_y = np.clip(fg_y, 0, img.shape[1] - w)
        cii = np.clip(cii, 0, img.shape[0] - w)
        cjj = np.clip(cjj, 0, img.shape[1] - w)

        for i in range(fg_indices.shape[0]):
            img_recon[fg_x[i]:fg_x[i] + w, fg_y[i]:fg_y[i] + w, :] += im_bg[cii[i]:cii[i] + w, cjj[i]:cjj[i] + w, :]
            counts[fg_x[i]:fg_x[i] + w, fg_y[i]:fg_y[i] + w] += 1

        counts = np.expand_dims(counts, axis=-1)
        img_recon = np.round(img_recon / counts).astype(np.uint8)

        return img_recon

    def _compute_delta_e2000_matrix(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """
        Computes the perceptual color difference (ΔE 2000) between two images.

        Parameters:
        - img1: np.ndarray, the first input image (height x width x 3), expected in BGR format.
        - img2: np.ndarray, the second input image (height x width x 3), expected in BGR format.

        Returns:
        - similarity_matrix: np.ndarray, a matrix representing the similarity between img1 and img2,
        with values in the range [0,1] (higher values indicate greater similarity).

        Process:
        1. Convert OpenCV's default BGR images to RGB.
        2. Convert RGB images to the Lab color space for perceptual color comparison.
        3. Compute the pixel-wise ΔE 2000 color difference between the two images.
        4. Normalize the ΔE 2000 values to [0,1] for similarity representation.
        """
        # Convert BGR to RGB
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        # Convert RGB to Lab Color Space
        lab1 = rgb2lab(img1_rgb)
        lab2 = rgb2lab(img2_rgb)

        # Compute ΔE 2000 Color Difference
        delta_e_matrix = deltaE_ciede2000(lab1, lab2)

        # Normalize ΔE 2000 Values to [0,1]
        similarity_matrix = 1 - np.clip(delta_e_matrix / 100, 0, 1)

        return similarity_matrix

#################################################################################

class ContextMeasure:
    def __init__(self, beta2: float = 1.2, alpha: float = 5.0, gamma: int = 8, lambda_spatial: float = 20):
        """
        Initializes the ContextMeasure instance.

        Parameters:
            beta2 (float): Balancing factor for transmission and completeness.
            alpha (float): Scaling factor for Gaussian covariance.
            gamma (int): 
            lambda_spatial (float): 
        """
        self.beta2 = beta2
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_spatial = lambda_spatial
        self._exp_factor = math.e / (math.e - 1)


    def compute(self, fm: np.ndarray, gt: np.ndarray, img: np.ndarray = None) -> float:
        """
        Computes the context measure between foreground map and ground truth.

        Parameters:
            fm (numpy.ndarray): foreground map (values between 0 and 255).
            gt (numpy.ndarray): ground truth map (values between 0 and 255).

        Returns:
            float: context measure value.
        """
        X = self._preprocess_map(fm, binary_flag = False)
        Y = self._preprocess_map(gt, binary_flag = True)
        cov_matrix, x_dis, y_dis = self._compute_y_params(Y)
        K = self._gaussian_kernel(x_dis, y_dis, cov_matrix)

        # CIR
        cir = self._calculate_cir(X, Y, K)
        mcir = np.sum(cir * X) / (np.sum(X) + 1e-8)

        # CAC
        cac = self._calculate_cac(X, Y, K)
        if img is not None:
            _, cd = self._calculate_camouflage_degree(img, Y)
        else:
            cd = np.zeros_like(Y)
        wcac = np.sum(cac * (Y + cd)) / (np.sum(Y) + np.sum(cd) + 1e-8)

        return (1 + self.beta2) * mcir * wcac / (self.beta2 * mcir + wcac + 1e-8)

    def visualize(self, img: np.ndarray, gt: np.ndarray) -> tuple:
        Y = self._preprocess_map(gt, binary_flag = True)
        cov_matrix, x_dis, y_dis = self._compute_y_params(Y)
        kernel = self._gaussian_kernel(x_dis, y_dis, cov_matrix)
        img_recon, cd = self._calculate_camouflage_degree(img, Y)
        return img_recon, cd

    def _calculate_cir(self, X: np.ndarray, Y: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        x_binary = (X > 0).astype(int)
        global_relevance_matrix = cv2.filter2D(Y, cv2.CV_32F, kernel)
        return x_binary * global_relevance_matrix

    def _calculate_cac(self, X: np.ndarray, Y: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        X = X.astype(float)
        non_global_completeness_matrix = np.exp(-1 * cv2.filter2D(X, -1, kernel))
        global_completeness_matrix = 1 - non_global_completeness_matrix
        cac = self._exp_factor * Y * global_completeness_matrix
        return cac

    def _preprocess_map(self, img, binary_flag) -> np.ndarray:
        """
        Preprocesses the input map: converts to grayscale (normalizes fm, or binarizes gt).

        Parameters:
            img (numpy.ndarray): grayscale 0~255.
            binary_flag: True or False

        Returns:
            np.ndarray: (normalized grayscale fm, binarized gt)
        """
        # check img if it's a 2-channel image
        if img.ndim == 2:
            # Convert fm to grayscale
            if not binary_flag:
                img = img.astype(np.float64) / 255.0
            else:
            # Convert gt to grayscale 
                img = (img >= 128).astype(np.float64)  # binarize to 0 or 1

            return img
        else:
            print('img wrong')
            return None

    def _gaussian_kernel(self, x_dis: int, y_dis: int, cov_matrix: np.ndarray) -> np.ndarray:
        det_sigma = np.linalg.det(cov_matrix)
        inv_sigma = np.linalg.inv(cov_matrix)

        x, y = np.meshgrid(np.arange(-x_dis, x_dis + 1),
                           np.arange(-y_dis, y_dis + 1), indexing='ij')
        Z = np.stack([x, y], axis=-1)
        exp_term = np.einsum('...i,ij,...j->...', Z, inv_sigma, Z)

        kernel = np.exp(-0.5 * exp_term) / (2 * np.pi * np.sqrt(det_sigma))
        return kernel / np.sum(kernel)

    def _compute_y_params(self, Y: np.ndarray) -> tuple:
        points = np.argwhere(Y > 0)
        if len(points) <= 1:
            return np.diag([0.25, 0.25]), 1, 1

        cov_matrix = np.cov(points, rowvar=False)
        sigma_x = np.sqrt(cov_matrix[0, 0])
        sigma_y = np.sqrt(cov_matrix[1, 1])
        total_sigma = np.sqrt(cov_matrix[0, 0] + cov_matrix[1, 1])

        std_cov_matrix = self.alpha ** 2 * cov_matrix / (total_sigma ** 2)
        std_sigma_x = self.alpha * sigma_x / total_sigma
        std_sigma_y = self.alpha * sigma_y / total_sigma
        x_dis = round(3 * std_sigma_x)
        y_dis = round(3 * std_sigma_y)

        return std_cov_matrix, x_dis, y_dis

    def _calculate_camouflage_degree(self, img: np.ndarray, mask: np.ndarray, w: int = 7) -> tuple:
        """
        Computes the camouflage degree matrix with Lab+spatial ANN and RGB reconstruction.

        Parameters:
        - img: BGR image (H x W x 3)
        - mask: binary mask (H x W)
        - w: patch size

        Returns:
        - img_recon: RGB reconstructed image
        - camouflage_degree_matrix: camouflage map (H x W)
        """

        mask_binary = (mask > 0).astype(np.uint8)
        fg_mask = mask_binary
        bg_mask = self._extract_surrounding_background(fg_mask, kernel_size=20)
        im_fg = fg_mask[:, :, np.newaxis] * img
        im_bg = bg_mask[:, :, np.newaxis] * img
        im_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        # Step 1: from Lab space to extract patch
        im_fg_lab = im_lab * fg_mask[:, :, np.newaxis]
        im_bg_lab = im_lab * bg_mask[:, :, np.newaxis]

        fg_indices, fg_feat_lab = self._extract_patches(im_fg_lab, fg_mask, w, d=w // 2)
        bg_indices, bg_feat_lab = self._extract_patches(im_bg_lab, bg_mask, w, d=w // 2)

        # Step 2: Lab+spatial for ANN query
        fg_nn = self._ann_with_spatial_faiss(
            bg_feat_lab, fg_feat_lab,
            bg_indices, fg_indices
        )

        # Step 3: reconstruct foreground in RGB space
        img_recon = self._reconstruct_image(
            img, fg_indices, bg_indices, fg_nn, im_bg, w
        )

        # Step 4: compute similarity in Lab space
        similarity_matrix = self._compute_delta_e2000_matrix(
            img_recon, im_fg.astype(np.uint8)
        ).astype(np.float64)

        # Step 5: compute Camouflage Degree
        cd = (
            (np.exp(self.gamma * similarity_matrix * mask_binary) - 1) / (np.exp(self.gamma) - 1)
        ).astype(np.float64)

        return img_recon, cd


    def _ann_with_spatial(self, x: np.ndarray, q: np.ndarray, x_coords: np.ndarray, q_coords: np.ndarray, m: int = 16) -> np.ndarray:
        """
        HNSWLib-based ANN search with spatial-aware feature augmentation.
        """
        # 标准化坐标
        scaler = StandardScaler()
        all_coords = np.vstack([x_coords, q_coords])
        scaled_coords = scaler.fit_transform(all_coords)
        x_coords_scaled = scaled_coords[:len(x_coords)]
        q_coords_scaled = scaled_coords[len(x_coords):]

        # 拼接图像特征 + 空间特征
        x_aug = np.hstack([x, self.lambda_spatial * x_coords_scaled])
        q_aug = np.hstack([q, self.lambda_spatial * q_coords_scaled])

        x_aug = x_aug.astype(np.float32)
        q_aug = q_aug.astype(np.float32)

        dim = x_aug.shape[1]
        num_elements = x_aug.shape[0]

        # 初始化 HNSW Index
        index = hnswlib.Index(space='l2', dim=dim)
        index.init_index(max_elements=num_elements, ef_construction=200, M=m, random_seed=42)
        index.add_items(x_aug)
        index.set_ef(64)

        # 查询最近邻
        k = 1
        indices, _ = index.knn_query(q_aug, k)
        return indices

    def _ann_with_spatial_faiss(self, x, q, x_coords, q_coords, m=16):
        # 坐标标准化 + 空间增强
        scaler = StandardScaler()
        all_coords = np.vstack([x_coords, q_coords])
        scaled_coords = scaler.fit_transform(all_coords)
        x_coords_scaled = scaled_coords[:len(x_coords)]
        q_coords_scaled = scaled_coords[len(x_coords):]

        x_aug = np.hstack([x, self.lambda_spatial * x_coords_scaled]).astype(np.float32)
        q_aug = np.hstack([q, self.lambda_spatial * q_coords_scaled]).astype(np.float32)

        # 建立 Index
        dim = x_aug.shape[1]
        index = faiss.IndexFlatL2(dim)  # L2 距离
        index.add(x_aug)

        _, indices = index.search(q_aug, 1)  # top-1
        return indices

    def _extract_surrounding_background(self, mask: np.ndarray, kernel_size: int = 50) -> np.ndarray:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        surrounding_bg_mask = dilated_mask - mask
        return surrounding_bg_mask

    def _extract_patches(self, img: np.ndarray, mask: np.ndarray, w: int, d: int) -> tuple:
        h, w_, c = img.shape
        pad_h = (d - (h - w) % d) % d
        pad_w = (d - (w_ - w) % d) % d
        img_padded = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        mask_padded = np.pad(mask, ((0, pad_h), (0, pad_w)), mode='constant')

        new_h, new_w = img_padded.shape[:2]

        img_patches = np.lib.stride_tricks.sliding_window_view(
            img_padded, (w, w, img.shape[2])
        )[::d, ::d, 0, :, :, :]
        mask_patches = np.lib.stride_tricks.sliding_window_view(
            mask_padded, (w, w)
        )[::d, ::d, :, :]

        img_patches = img_patches.reshape(-1, w * w * c)
        mask_patches = mask_patches.reshape(-1, w, w)

        grid_x, grid_y = np.meshgrid(
            np.arange(0, new_h - w + 1, d),
            np.arange(0, new_w - w + 1, d),
            indexing='ij'
        )
        all_indices = np.column_stack((grid_x.ravel(), grid_y.ravel()))
        valid_idx = np.all(mask_patches > 0, axis=(1, 2))
        valid_indices = all_indices[valid_idx]
        valid_patches = img_patches[valid_idx]

        return valid_indices, valid_patches

    def _reconstruct_image(self, img: np.ndarray, fg_indices: np.ndarray, bg_indices: np.ndarray, fg_nn: np.ndarray, im_bg: np.ndarray, w: int) -> np.ndarray:
        img_recon = np.zeros_like(img, dtype=np.int64)
        counts = np.zeros(img.shape[:2]) + 1e-8

        fg_x, fg_y = fg_indices[:, 0], fg_indices[:, 1]
        nn_i_j = fg_nn[:, 0]
        cii, cjj = bg_indices[nn_i_j, 0], bg_indices[nn_i_j, 1]

        fg_x = np.clip(fg_x, 0, img.shape[0] - w)
        fg_y = np.clip(fg_y, 0, img.shape[1] - w)
        cii = np.clip(cii, 0, img.shape[0] - w)
        cjj = np.clip(cjj, 0, img.shape[1] - w)

        for i in range(fg_indices.shape[0]):
            img_recon[fg_x[i]:fg_x[i] + w, fg_y[i]:fg_y[i] + w, :] += im_bg[cii[i]:cii[i] + w, cjj[i]:cjj[i] + w, :]
            counts[fg_x[i]:fg_x[i] + w, fg_y[i]:fg_y[i] + w] += 1

        counts = np.expand_dims(counts, axis=-1)
        img_recon = np.round(img_recon / counts).astype(np.uint8)

        return img_recon

    def _compute_delta_e2000_matrix(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """
        Computes the perceptual color difference (ΔE 2000) between two images.

        Parameters:
        - img1: np.ndarray, the first input image (height x width x 3), expected in BGR format.
        - img2: np.ndarray, the second input image (height x width x 3), expected in BGR format.

        Returns:
        - similarity_matrix: np.ndarray, a matrix representing the similarity between img1 and img2,
        with values in the range [0,1] (higher values indicate greater similarity).

        Process:
        1. Convert OpenCV's default BGR images to RGB.
        2. Convert RGB images to the Lab color space for perceptual color comparison.
        3. Compute the pixel-wise ΔE 2000 color difference between the two images.
        4. Normalize the ΔE 2000 values to [0,1] for similarity representation.
        """
        # Convert BGR to RGB
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        # Convert RGB to Lab Color Space
        lab1 = rgb2lab(img1_rgb)
        lab2 = rgb2lab(img2_rgb)

        # Compute ΔE 2000 Color Difference
        delta_e_matrix = deltaE_ciede2000(lab1, lab2)

        # Normalize ΔE 2000 Values to [0,1]
        similarity_matrix = 1 - np.clip(delta_e_matrix / 100, 0, 1)

        return similarity_matrix
