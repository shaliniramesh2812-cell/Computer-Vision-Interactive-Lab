# ============================================================
# COMPUTER VISION INTERACTIVE LAB
# ============================================================

import os

# Must be set BEFORE TensorFlow / DeepFace imports
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tempfile
import traceback

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from PIL import Image, ImageOps


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Computer Vision Interactive Lab",
    page_icon="👁️",
    layout="wide"
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
    }

    .subtitle {
        font-size: 18px;
        color: #666;
        margin-bottom: 25px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">👁️ Computer Vision Interactive Lab</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Upload images, select computer vision techniques,
    compare results, and understand the calculations.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# IMAGE FUNCTIONS
# ============================================================

def load_image(uploaded_file):
    """
    Load image and fix EXIF orientation.
    """

    image = Image.open(uploaded_file)

    image = ImageOps.exif_transpose(image)

    return image.convert("RGB")


def pil_to_bgr(image):
    rgb = np.array(image)

    return cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR
    )


def bgr_to_rgb(image):
    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


# ============================================================
# RESIZE FOR DEEPFACE
# ============================================================

def prepare_image_for_deepface(
    image,
    max_dimension=1600
):
    """
    Resize very large DSLR images.

    This prevents DeepFace from processing
    unnecessarily huge 5000-7000 pixel images.
    """

    image = image.copy()

    width, height = image.size

    largest_dimension = max(
        width,
        height
    )

    if largest_dimension <= max_dimension:
        return image

    scale = (
        max_dimension /
        float(largest_dimension)
    )

    new_width = int(
        width * scale
    )

    new_height = int(
        height * scale
    )

    resized = image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )

    return resized


# ============================================================
# SAVE TEMP IMAGE
# ============================================================

def save_image_for_deepface(
    image,
    prefix
):
    """
    Save optimized image to temporary JPG.
    """

    optimized = prepare_image_for_deepface(
        image,
        max_dimension=1600
    )

    temp = tempfile.NamedTemporaryFile(
        suffix=".jpg",
        prefix=prefix + "_",
        delete=False
    )

    path = temp.name

    temp.close()

    optimized.save(
        path,
        format="JPEG",
        quality=95
    )

    return path


def remove_temp_file(path):

    try:

        if path and os.path.exists(path):
            os.remove(path)

    except Exception:
        pass


# ============================================================
# HAAR CASCADE
# ============================================================

@st.cache_resource
def load_haar():

    path = (
        cv2.data.haarcascades
        +
        "haarcascade_frontalface_default.xml"
    )

    cascade = cv2.CascadeClassifier(path)

    if cascade.empty():

        raise RuntimeError(
            "Haar cascade could not be loaded."
        )

    return cascade


# ============================================================
# VIOLA-JONES
# ============================================================

def run_viola_jones(
    image,
    min_neighbors=8,
    single_person=False
):

    image_bgr = pil_to_bgr(image)

    gray = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.equalizeHist(gray)

    cascade = load_haar()

    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.15,
        minNeighbors=min_neighbors,
        minSize=(70, 70)
    )

    boxes = []

    for x, y, w, h in faces:

        area = w * h

        image_area = (
            image_bgr.shape[0]
            *
            image_bgr.shape[1]
        )

        area_ratio = (
            area /
            image_area
        )

        aspect_ratio = (
            w /
            float(h)
        )

        if area_ratio < 0.005:
            continue

        if aspect_ratio < 0.55:
            continue

        if aspect_ratio > 1.8:
            continue

        boxes.append(
            (
                int(x),
                int(y),
                int(w),
                int(h)
            )
        )

    # Remove duplicate/overlapping detections
    final_boxes = []

    boxes = sorted(
        boxes,
        key=lambda b: b[2] * b[3],
        reverse=True
    )

    for box in boxes:

        x1, y1, w1, h1 = box

        keep = True

        for other in final_boxes:

            x2, y2, w2, h2 = other

            xa = max(x1, x2)
            ya = max(y1, y2)

            xb = min(
                x1 + w1,
                x2 + w2
            )

            yb = min(
                y1 + h1,
                y2 + h2
            )

            intersection = max(
                0,
                xb - xa
            ) * max(
                0,
                yb - ya
            )

            area1 = w1 * h1
            area2 = w2 * h2

            union = (
                area1
                +
                area2
                -
                intersection
            )

            iou = (
                intersection / union
                if union > 0
                else 0
            )

            if iou > 0.35:
                keep = False
                break

        if keep:
            final_boxes.append(box)

    if single_person and final_boxes:

        final_boxes = [
            max(
                final_boxes,
                key=lambda b: b[2] * b[3]
            )
        ]

    output = image_bgr.copy()

    for i, (x, y, w, h) in enumerate(
        final_boxes
    ):

        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            3
        )

        cv2.putText(
            output,
            f"Face {i + 1}",
            (x, max(30, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    return {
        "image": bgr_to_rgb(output),
        "boxes": final_boxes,
        "count": len(final_boxes)
    }


# ============================================================
# TEMPLATE MATCHING
# ============================================================

def run_template_matching(
    input_image,
    reference_image
):

    input_bgr = pil_to_bgr(
        input_image
    )

    reference_bgr = pil_to_bgr(
        reference_image
    )

    input_gray = cv2.cvtColor(
        input_bgr,
        cv2.COLOR_BGR2GRAY
    )

    reference_gray = cv2.cvtColor(
        reference_bgr,
        cv2.COLOR_BGR2GRAY
    )

    # If reference is larger than input,
    # resize reference.
    if (
        reference_gray.shape[0]
        >
        input_gray.shape[0]
        or
        reference_gray.shape[1]
        >
        input_gray.shape[1]
    ):

        scale = min(
            input_gray.shape[1]
            /
            reference_gray.shape[1],

            input_gray.shape[0]
            /
            reference_gray.shape[0]
        ) * 0.8

        new_width = max(
            20,
            int(
                reference_gray.shape[1]
                * scale
            )
        )

        new_height = max(
            20,
            int(
                reference_gray.shape[0]
                * scale
            )
        )

        reference_gray = cv2.resize(
            reference_gray,
            (
                new_width,
                new_height
            )
        )

    result = cv2.matchTemplate(
        input_gray,
        reference_gray,
        cv2.TM_CCOEFF_NORMED
    )

    _, max_value, _, max_location = (
        cv2.minMaxLoc(result)
    )

    h, w = reference_gray.shape

    x, y = max_location

    output = input_bgr.copy()

    cv2.rectangle(
        output,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        3
    )

    return {
        "image": bgr_to_rgb(output),
        "score": float(max_value),
        "location": (x, y)
    }


# ============================================================
# DEEPFACE
# ============================================================

@st.cache_resource
def get_deepface():

    from deepface import DeepFace

    return DeepFace


# ============================================================
# DEEPFACE FACE DETECTION TEST
# ============================================================

def test_deepface_detection(
    image_path,
    detector
):

    DeepFace = get_deepface()

    faces = DeepFace.extract_faces(
        img_path=image_path,
        detector_backend=detector,
        enforce_detection=True,
        align=True,
        anti_spoofing=False
    )

    return faces


# ============================================================
# DRAW DEEPFACE FACE
# ============================================================

def draw_face_area(
    image,
    area,
    label="Face"
):

    output = pil_to_bgr(
        image
    )

    if not area:
        return bgr_to_rgb(output)

    x = area.get("x")
    y = area.get("y")
    w = area.get("w")
    h = area.get("h")

    if None in (x, y, w, h):

        return bgr_to_rgb(output)

    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)

    cv2.rectangle(
        output,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        4
    )

    cv2.putText(
        output,
        label,
        (x, max(30, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    return bgr_to_rgb(output)


# ============================================================
# ROBUST DEEPFACE VERIFICATION
# ============================================================

def verify_with_deepface(
    input_image,
    reference_image,
    model_name,
    normalization
):

    DeepFace = get_deepface()

    input_path = None
    reference_path = None

    try:

        # ----------------------------------------------------
        # Resize huge images
        # ----------------------------------------------------

        input_path = save_image_for_deepface(
            input_image,
            "input"
        )

        reference_path = save_image_for_deepface(
            reference_image,
            "reference"
        )

        # ----------------------------------------------------
        # Detector order
        # ----------------------------------------------------
        #
        # OpenCV is lightweight.
        #
        # RetinaFace is used as a fallback if available.
        #
        # ----------------------------------------------------

        detectors = [
            "opencv",
            "retinaface"
        ]

        errors = []

        for detector in detectors:

            try:

                # First explicitly test detection.
                # This gives us a clearer failure point.

                faces1 = test_deepface_detection(
                    input_path,
                    detector
                )

                faces2 = test_deepface_detection(
                    reference_path,
                    detector
                )

                if len(faces1) == 0:
                    raise ValueError(
                        "No face detected in input image."
                    )

                if len(faces2) == 0:
                    raise ValueError(
                        "No face detected in reference image."
                    )

                # ------------------------------------------------
                # Actual verification
                # ------------------------------------------------

                result = DeepFace.verify(

                    img1_path=input_path,

                    img2_path=reference_path,

                    model_name=model_name,

                    detector_backend=detector,

                    distance_metric="cosine",

                    enforce_detection=True,

                    align=True,

                    normalization=normalization,

                    threshold=None,

                    silent=False
                )

                result["detector_used"] = detector

                result["input_face_count"] = len(
                    faces1
                )

                result["reference_face_count"] = len(
                    faces2
                )

                return result

            except Exception as detector_error:

                errors.append(
                    f"{detector}: {str(detector_error)}"
                )

                continue

        # ----------------------------------------------------
        # If every detector failed
        # ----------------------------------------------------

        raise RuntimeError(
            "All DeepFace face detectors failed.\n\n"
            +
            "\n".join(errors)
        )

    finally:

        remove_temp_file(
            input_path
        )

        remove_temp_file(
            reference_path
        )


# ============================================================
# ARCFACE
# ============================================================

def run_arcface(
    input_image,
    reference_image
):

    return verify_with_deepface(
        input_image=input_image,
        reference_image=reference_image,
        model_name="ArcFace",
        normalization="ArcFace"
    )


# ============================================================
# FACENET
# ============================================================

def run_facenet(
    input_image,
    reference_image
):

    return verify_with_deepface(
        input_image=input_image,
        reference_image=reference_image,
        model_name="Facenet",
        normalization="Facenet"
    )


# ============================================================
# DISPLAY DEEPFACE RESULT
# ============================================================

def display_deepface_result(
    result,
    input_image,
    reference_image,
    model_name
):

    verified = result.get(
        "verified",
        False
    )

    distance = result.get(
        "distance"
    )

    threshold = result.get(
        "threshold"
    )

    confidence = result.get(
        "confidence"
    )

    detector = result.get(
        "detector_used",
        result.get(
            "detector_backend",
            "Unknown"
        )
    )

    areas = result.get(
        "facial_areas",
        {}
    )

    # --------------------------------------------------------
    # Main result
    # --------------------------------------------------------

    if verified:

        st.success(
            f"✅ SAME PERSON — {model_name}"
        )

    else:

        st.error(
            f"❌ DIFFERENT PERSON — {model_name}"
        )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    st.markdown(
        "### 📊 Verification Metrics"
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Cosine Distance",
            (
                f"{float(distance):.4f}"
                if distance is not None
                else "N/A"
            )
        )

    with c2:

        st.metric(
            "Threshold",
            (
                f"{float(threshold):.4f}"
                if threshold is not None
                else "N/A"
            )
        )

    with c3:

        st.metric(
            "Confidence",
            (
                f"{float(confidence):.2f}%"
                if confidence is not None
                else "N/A"
            )
        )

    with c4:

        st.metric(
            "Detector",
            detector
        )

    # --------------------------------------------------------
    # Decision calculation
    # --------------------------------------------------------

    st.markdown(
        "### 🧮 Decision Calculation"
    )

    if (
        distance is not None
        and threshold is not None
    ):

        st.code(
            f"""
Cosine Distance = {float(distance):.6f}

Threshold       = {float(threshold):.6f}

Decision:

Distance <= Threshold
"""

        )

        if float(distance) <= float(threshold):

            st.success(
                f"{float(distance):.6f} <= "
                f"{float(threshold):.6f} "
                "→ SAME PERSON"
            )

        else:

            st.error(
                f"{float(distance):.6f} > "
                f"{float(threshold):.6f} "
                "→ DIFFERENT PERSON"
            )

    # --------------------------------------------------------
    # Face areas
    # --------------------------------------------------------

    if areas:

        st.markdown(
            "### 👤 Faces Used for Recognition"
        )

        col1, col2 = st.columns(2)

        area1 = areas.get(
            "img1"
        )

        area2 = areas.get(
            "img2"
        )

        with col1:

            img1 = draw_face_area(
                input_image,
                area1,
                "Input Face"
            )

            st.image(
                img1,
                caption="Input face selected by DeepFace",
                use_container_width=True
            )

        with col2:

            img2 = draw_face_area(
                reference_image,
                area2,
                "Reference Face"
            )

            st.image(
                img2,
                caption="Reference face selected by DeepFace",
                use_container_width=True
            )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "⚙️ Computer Vision Methods"
)

methods = st.sidebar.multiselect(

    "Select methods:",

    [
        "Template Matching",
        "Viola-Jones Face Detection",
        "DeepFace - ArcFace",
        "DeepFace - FaceNet"
    ],

    default=[
        "DeepFace - ArcFace"
    ]
)


# ============================================================
# VIOLA SETTINGS
# ============================================================

if "Viola-Jones Face Detection" in methods:

    st.sidebar.markdown(
        "---"
    )

    min_neighbors = st.sidebar.slider(
        "Minimum Neighbors",
        5,
        12,
        8
    )

    single_person = st.sidebar.checkbox(
        "Single Person Mode",
        False
    )

else:

    min_neighbors = 8
    single_person = False


# ============================================================
# UPLOAD
# ============================================================

st.markdown(
    "## 📷 Upload Images"
)

col1, col2 = st.columns(2)

with col1:

    input_file = st.file_uploader(
        "Input Image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ]
    )

with col2:

    if any(
        x in methods
        for x in [
            "Template Matching",
            "DeepFace - ArcFace",
            "DeepFace - FaceNet"
        ]
    ):

        reference_file = st.file_uploader(
            "Reference Image",
            type=[
                "jpg",
                "jpeg",
                "png",
                "webp"
            ]
        )

    else:

        reference_file = None


# ============================================================
# LOAD IMAGES
# ============================================================

input_image = None
reference_image = None

if input_file:

    input_image = load_image(
        input_file
    )

if reference_file:

    reference_image = load_image(
        reference_file
    )


# ============================================================
# DISPLAY
# ============================================================

if input_image:

    st.markdown(
        "### Uploaded Images"
    )

    c1, c2 = st.columns(2)

    with c1:

        st.image(
            input_image,
            caption="Input Image",
            use_container_width=True
        )

    with c2:

        if reference_image:

            st.image(
                reference_image,
                caption="Reference Image",
                use_container_width=True
            )


# ============================================================
# VALIDATION
# ============================================================

if not methods:

    st.info(
        "Select at least one method."
    )

    st.stop()


if input_image is None:

    st.warning(
        "Please upload an input image."
    )

    st.stop()


reference_needed = any(
    x in methods
    for x in [
        "Template Matching",
        "DeepFace - ArcFace",
        "DeepFace - FaceNet"
    ]
)

if (
    reference_needed
    and
    reference_image is None
):

    st.warning(
        "Please upload a reference image."
    )

    st.stop()


# ============================================================
# RUN
# ============================================================

st.markdown(
    "---"
)

if st.button(
    "🚀 Run Computer Vision",
    type="primary",
    use_container_width=True
):

    summary = []

    st.markdown(
        "## 🔍 Results"
    )

    # ========================================================
    # TEMPLATE MATCHING
    # ========================================================

    if "Template Matching" in methods:

        st.markdown(
            "## 🎯 Template Matching"
        )

        with st.spinner(
            "Running Template Matching..."
        ):

            try:

                result = run_template_matching(
                    input_image,
                    reference_image
                )

                st.image(
                    result["image"],
                    caption="Template Matching",
                    use_container_width=True
                )

                score = result["score"]

                st.metric(
                    "Match Score",
                    f"{score:.4f}"
                )

                st.markdown(
                    """
                    Template Matching compares the reference
                    template against regions of the input image.

                    `TM_CCOEFF_NORMED`

                    Higher values indicate stronger correlation.
                    """
                )

                summary.append(
                    {
                        "Method": "Template Matching",
                        "Result": "Completed",
                        "Score": round(
                            score,
                            4
                        )
                    }
                )

            except Exception as e:

                st.error(
                    f"Template Matching failed: {e}"
                )


    # ========================================================
    # VIOLA-JONES
    # ========================================================

    if "Viola-Jones Face Detection" in methods:

        st.markdown(
            "## 👤 Viola–Jones Face Detection"
        )

        with st.spinner(
            "Detecting faces..."
        ):

            try:

                result = run_viola_jones(
                    input_image,
                    min_neighbors,
                    single_person
                )

                st.image(
                    result["image"],
                    caption="Viola–Jones Result",
                    use_container_width=True
                )

                st.metric(
                    "Faces Detected",
                    result["count"]
                )

                st.markdown(
                    """
                    Viola–Jones uses Haar-like features,
                    integral images, AdaBoost and a cascade
                    classifier for face detection.
                    """
                )

                summary.append(
                    {
                        "Method": "Viola-Jones",
                        "Result": f"{result['count']} face(s)",
                        "Score": result["count"]
                    }
                )

            except Exception as e:

                st.error(
                    f"Viola-Jones failed: {e}"
                )


    # ========================================================
    # ARCFACE
    # ========================================================

    if "DeepFace - ArcFace" in methods:

        st.markdown(
            "## 🧠 DeepFace + ArcFace"
        )

        st.info(
            "Large images are automatically resized for "
            "face recognition. DeepFace performs its own "
            "face detection and alignment."
        )

        with st.spinner(
            "Running ArcFace..."
        ):

            try:

                result = run_arcface(
                    input_image,
                    reference_image
                )

                display_deepface_result(
                    result,
                    input_image,
                    reference_image,
                    "ArcFace"
                )

                summary.append(
                    {
                        "Method": "ArcFace",
                        "Result": (
                            "SAME PERSON"
                            if result.get(
                                "verified",
                                False
                            )
                            else "DIFFERENT PERSON"
                        ),
                        "Score": round(
                            float(
                                result.get(
                                    "distance",
                                    0
                                )
                            ),
                            4
                        )
                    }
                )

            except Exception as e:

                st.error(
                    "❌ ArcFace could not process the images."
                )

                st.markdown(
                    "### 🔎 Actual Technical Error"
                )

                st.code(
                    str(e)
                )

                # Show full underlying exception
                cause = getattr(
                    e,
                    "__cause__",
                    None
                )

                if cause:

                    st.markdown(
                        "### Underlying DeepFace Error"
                    )

                    st.code(
                        str(cause)
                    )

                st.info(
                    """
                    The application did not hide the real
                    detector/model error. The message above
                    tells us whether the problem is face
                    detection, model loading, image format,
                    or another DeepFace issue.
                    """
                )


    # ========================================================
    # FACENET
    # ========================================================

    if "DeepFace - FaceNet" in methods:

        st.markdown(
            "## 🔵 DeepFace + FaceNet"
        )

        with st.spinner(
            "Running FaceNet..."
        ):

            try:

                result = run_facenet(
                    input_image,
                    reference_image
                )

                display_deepface_result(
                    result,
                    input_image,
                    reference_image,
                    "FaceNet"
                )

                summary.append(
                    {
                        "Method": "FaceNet",
                        "Result": (
                            "SAME PERSON"
                            if result.get(
                                "verified",
                                False
                            )
                            else "DIFFERENT PERSON"
                        ),
                        "Score": round(
                            float(
                                result.get(
                                    "distance",
                                    0
                                )
                            ),
                            4
                        )
                    }
                )

            except Exception as e:

                st.error(
                    "❌ FaceNet could not process the images."
                )

                st.code(
                    str(e)
                )

                cause = getattr(
                    e,
                    "__cause__",
                    None
                )

                if cause:

                    st.markdown(
                        "### Underlying Error"
                    )

                    st.code(
                        str(cause)
                    )


    # ========================================================
    # SUMMARY
    # ========================================================

    if summary:

        st.markdown(
            "---"
        )

        st.markdown(
            "## 📊 Comparison"
        )

        df = pd.DataFrame(
            summary
        )

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# EDUCATIONAL SECTION
# ============================================================

st.markdown(
    "---"
)

st.markdown(
    "## 📚 How the Methods Work"
)

with st.expander(
    "🎯 Template Matching"
):

    st.write(
        """
        Template Matching searches for a reference image
        inside an input image using correlation.
        """
    )

with st.expander(
    "👤 Viola–Jones"
):

    st.write(
        """
        Viola–Jones uses Haar features, integral images,
        AdaBoost and cascade classifiers for face detection.
        """
    )

with st.expander(
    "🧠 ArcFace"
):

    st.write(
        """
        ArcFace converts a detected and aligned face into
        a numerical embedding.

        Two embeddings are compared using cosine distance.

        Lower cosine distance means greater similarity.

        DeepFace uses a model-specific tuned threshold
        to make the SAME PERSON / DIFFERENT PERSON decision.
        """
    )

with st.expander(
    "🔵 FaceNet"
):

    st.write(
        """
        FaceNet maps faces into an embedding space where
        images of the same person should be close together
        and images of different people should be farther apart.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    "---"
)

st.caption(
    "Computer Vision Interactive Lab | "
    "OpenCV + DeepFace + Streamlit"
)