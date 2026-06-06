import pytest

from llmrun.attachments import build_attachments


def test_build_local_image_attachment_encodes_data_url(tmp_path):
    image = tmp_path / "screenshot.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")

    attachments = build_attachments(images=[str(image)], files=[], image_detail="high")

    assert len(attachments) == 1
    attachment = attachments[0]
    assert attachment.kind == "image"
    assert attachment.source == str(image)
    assert attachment.filename == "screenshot.png"
    assert attachment.mime_type == "image/png"
    assert attachment.payload == "data:image/png;base64,iVBORw0KGgo="
    assert attachment.detail == "high"


def test_build_url_attachments_keep_url_without_payload():
    attachments = build_attachments(
        images=["https://example.test/image.jpg"],
        files=["https://example.test/report.pdf"],
        image_detail="auto",
    )

    assert attachments[0].payload is None
    assert attachments[0].source == "https://example.test/image.jpg"
    assert attachments[0].filename == "image.jpg"
    assert attachments[1].payload is None
    assert attachments[1].mime_type == "application/pdf"


def test_https_url_without_filename_uses_default_name():
    attachments = build_attachments(
        images=["https://example.test/"],
        files=["https://example.test"],
        image_detail="low",
    )

    assert attachments[0].filename == "image"
    assert attachments[0].mime_type == "image/png"
    assert attachments[1].filename == "file"
    assert attachments[1].mime_type == "application/octet-stream"


def test_unknown_local_file_mime_type_uses_kind_default(tmp_path):
    image = tmp_path / "image.unknownext"
    data = tmp_path / "data.unknownext"
    image.write_bytes(b"image")
    data.write_bytes(b"data")

    attachments = build_attachments(
        images=[str(image)], files=[str(data)], image_detail="auto"
    )

    assert attachments[0].mime_type == "image/png"
    assert attachments[1].mime_type == "application/octet-stream"


def test_build_attachment_rejects_missing_local_path():
    with pytest.raises(ValueError, match="Attachment does not exist"):
        build_attachments(images=["missing.png"], files=[], image_detail="auto")


def test_build_attachment_rejects_non_https_urls_and_directories(tmp_path):
    directory = tmp_path / "folder"
    directory.mkdir()

    with pytest.raises(ValueError, match="must use https"):
        build_attachments(
            images=["http://example.test/image.png"], files=[], image_detail="auto"
        )
    with pytest.raises(ValueError, match="not a file"):
        build_attachments(images=[str(directory)], files=[], image_detail="auto")
