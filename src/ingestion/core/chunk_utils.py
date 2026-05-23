import re


def clean_mdx(content: str) -> str:
    content = re.sub(
        r"<h([1-6]).*?>(.*?)</h\1>",
        lambda m: "#" * int(m.group(1)) + " " + m.group(2),
        content,
        flags=re.DOTALL,
    )

    content = re.sub(
        r'<Card title="([^"]+)".*?>\s*(.*?)\s*</Card>',
        r"### \1\n\2",
        content,
        flags=re.DOTALL,
    )

    content = re.sub(r"<[^>]+>", "", content)

    return content
