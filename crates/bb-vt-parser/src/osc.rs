//! OSC (Operating System Command) payload parsing.

use crate::VtEvent;

/// Parse a complete OSC payload (everything between `ESC ]` and the
/// terminator) and push any resulting semantic event.
///
/// Recognized codes: 0/2 (title), 7 (working directory), 8 (hyperlink),
/// 133 (prompt protocol). Everything else is ignored. The payload is
/// decoded as UTF-8 with invalid bytes replaced by U+FFFD.
pub(crate) fn dispatch(buf: &[u8], events: &mut Vec<VtEvent>) {
    let s = String::from_utf8_lossy(buf);
    let (code, rest) = match s.split_once(';') {
        Some((code, rest)) => (code, Some(rest)),
        None => (s.as_ref(), None),
    };
    let Ok(code) = code.parse::<u32>() else {
        return;
    };
    match code {
        0 | 2 => events.push(VtEvent::Title(rest.unwrap_or("").to_owned())),
        7 => events.push(VtEvent::WorkingDirectory(rest.unwrap_or("").to_owned())),
        8 => hyperlink(rest, events),
        133 => prompt_marker(rest, events),
        _ => {} // unknown OSC: ignored
    }
}

/// OSC 8 — `8;params;uri`. An empty URI closes the hyperlink.
fn hyperlink(rest: Option<&str>, events: &mut Vec<VtEvent>) {
    let Some(rest) = rest else {
        return; // "8" with no payload: malformed, ignored
    };
    let Some((_params, uri)) = rest.split_once(';') else {
        return; // missing the params/uri separator: malformed, ignored
    };
    let link = if uri.is_empty() {
        None
    } else {
        Some(uri.to_owned())
    };
    events.push(VtEvent::Hyperlink(link));
}

/// OSC 133 — the shell-integration prompt protocol (`A`/`B`/`C`/`D`).
fn prompt_marker(rest: Option<&str>, events: &mut Vec<VtEvent>) {
    let Some(rest) = rest else {
        return; // bare "133": ignored
    };
    let (cmd, arg) = match rest.split_once(';') {
        Some((cmd, arg)) => (cmd, Some(arg)),
        None => (rest, None),
    };
    match cmd {
        "A" => events.push(VtEvent::PromptStart),
        "B" => events.push(VtEvent::CommandStart),
        "C" => events.push(VtEvent::CommandExecuted),
        "D" => {
            // The exit code may be followed by extra `;key=value` fields
            // (as some shells send); only the first segment is the code.
            let exit_code = arg
                .and_then(|a| a.split(';').next())
                .and_then(|c| c.parse::<i32>().ok());
            events.push(VtEvent::CommandFinished { exit_code });
        }
        _ => {} // unknown 133 subcommand: ignored
    }
}
