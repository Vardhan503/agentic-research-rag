import re
from urllib.parse import urlparse


MDPI_HOSTS = {
    "www.mdpi.com",
    "mdpi.com",
}

DEFAULT_CDN_BASE_URL = (
    "https://mdpi-res.com/d_attachment"
)

# MDPI journal slugs that cannot be derived from the
# journal display name or the DOI short code.
KNOWN_MDPI_SLUGS = {
    "applied sciences": "applsci",
    "future internet": "futureinternet",
    "big data and cognitive computing": "BDCC",
    "isprs international journal of geo-information": "ijgi",
    "machine learning and knowledge extraction": "make",
    "journal of clinical medicine": "jcm",
    "education sciences": "education",
    "journal of imaging": "jimaging",
    "social sciences": "socsci",
    "journal of cybersecurity and privacy": "jcp",
    "trends in higher education": "higheredu",
    "european burn journal": "ebj",
    "multimodal technologies and interaction": "mti",
    "anesthesia research": "anesthres",
    "international journal of molecular sciences": "ijms",
    "international journal of environmental research and public health": "ijerph",
}


def is_mdpi_url(url):
    if not url:
        return False

    hostname = urlparse(str(url)).hostname

    if not hostname:
        return False

    return hostname.lower() in MDPI_HOSTS


def mdpi_article_parts(url):
    """
    Parse an MDPI article URL such as
    https://www.mdpi.com/2078-2489/17/2/133/pdf?version=1
    into (issn, volume, article_number).

    Returns None when the URL does not match.
    """
    if not is_mdpi_url(url):
        return None

    path = urlparse(str(url)).path

    segments = [
        segment
        for segment in path.split("/")
        if segment
    ]

    if len(segments) < 4:
        return None

    issn = segments[0]
    volume = segments[1]
    article = segments[3]

    if not re.fullmatch(
        r"\d{4}-\d{3}[\dXx]",
        issn,
    ):
        return None

    if not volume.isdigit():
        return None

    if not article.isdigit():
        return None

    return (
        issn,
        int(volume),
        int(article),
    )


def mdpi_doi_short_code(doi):
    if not doi:
        return ""

    match = re.search(
        r"10\.3390/([a-z]+)\d",
        str(doi).lower(),
    )

    if not match:
        return ""

    return match.group(1)


def journal_display_name(paper):
    primary_location = paper.get(
        "primary_location"
    ) or {}

    source = primary_location.get(
        "source"
    ) or {}

    return str(
        source.get("display_name") or ""
    ).strip()


def mdpi_slug_candidates(paper):
    """
    Ordered guesses for the MDPI journal slug used by
    the mdpi-res.com CDN path. Each guess must be
    verified with a request before it is trusted.
    """
    candidates = []

    display_name = journal_display_name(paper)
    normalized_name = display_name.lower()

    known_slug = KNOWN_MDPI_SLUGS.get(
        normalized_name
    )

    if known_slug:
        candidates.append(known_slug)

    if normalized_name:
        compact_name = re.sub(
            r"[^a-z0-9]+",
            "",
            normalized_name,
        )

        if compact_name:
            candidates.append(compact_name)

    doi_code = mdpi_doi_short_code(
        paper.get("doi")
    )

    if doi_code:
        candidates.append(doi_code)

    ordered = []

    for candidate in candidates:
        if candidate not in ordered:
            ordered.append(candidate)

    return ordered


def build_mdpi_cdn_url(
    slug,
    volume,
    article,
    base_url=DEFAULT_CDN_BASE_URL,
):
    # MDPI zero-pads the volume to two digits and the
    # article number to five, e.g. ai-06-00226.
    file_stem = (
        slug
        + "-"
        + str(volume).zfill(2)
        + "-"
        + str(article).zfill(5)
    )

    return (
        base_url.rstrip("/")
        + "/"
        + slug
        + "/"
        + file_stem
        + "/article_deploy/"
        + file_stem
        + ".pdf"
    )


class MdpiSlugResolver:
    """
    Verifies MDPI journal slugs against the CDN once
    per ISSN and remembers the outcome for the run.
    """

    def __init__(
        self,
        session,
        cdn_config,
        timeout,
    ):
        self.session = session
        self.base_url = cdn_config.get(
            "base_url",
            DEFAULT_CDN_BASE_URL,
        )
        self.timeout = timeout
        self.slug_by_issn = {}
        self.failed_issns = set()

    def candidate_works(self, url):
        response = None

        try:
            response = self.session.get(
                url,
                stream=True,
                allow_redirects=True,
                timeout=self.timeout,
            )

            if response.status_code >= 400:
                return False

            content_type = str(
                response.headers.get(
                    "Content-Type",
                    "",
                )
            ).lower()

            if (
                content_type
                and "pdf" not in content_type
                and "octet-stream" not in content_type
            ):
                return False

            return True

        except Exception:
            return False

        finally:
            if response is not None:
                try:
                    response.close()
                except Exception:
                    pass

    def resolve(self, paper, issn, volume, article):
        if issn in self.slug_by_issn:
            return self.slug_by_issn[issn]

        if issn in self.failed_issns:
            return None

        for slug in mdpi_slug_candidates(paper):
            url = build_mdpi_cdn_url(
                slug,
                volume,
                article,
                self.base_url,
            )

            if self.candidate_works(url):
                self.slug_by_issn[issn] = slug
                return slug

        self.failed_issns.add(issn)
        return None


def mdpi_cdn_url_for_paper(
    paper,
    resolver,
):
    """
    Return a verified mdpi-res.com PDF URL for the paper,
    or None when the paper is not an MDPI article or the
    journal slug could not be verified.
    """
    for url in paper.get("oa_pdf_urls") or []:
        parts = mdpi_article_parts(url)

        if parts is None:
            continue

        issn, volume, article = parts

        slug = resolver.resolve(
            paper,
            issn,
            volume,
            article,
        )

        if not slug:
            return None

        return build_mdpi_cdn_url(
            slug,
            volume,
            article,
            resolver.base_url,
        )

    return None
