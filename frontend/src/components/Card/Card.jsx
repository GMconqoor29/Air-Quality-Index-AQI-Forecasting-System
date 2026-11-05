/**
 * Card — Atmospheric telemetry module wrapper.
 * Supports optional module `code` (e.g. "MOD-01"), `subtitle`, `icon`, and `action`.
 */
export default function Card({
  title,
  subtitle,
  code,
  icon: Icon,
  action,
  children,
  className = '',
  accentColor,
}) {
  return (
    <section
      className={`card ${className}`.trim()}
      style={accentColor ? { '--card-accent': accentColor } : undefined}
    >
      {(title || action || code) && (
        <header className="card-header">
          <div className="card-header-left">
            <div className="card-title-row">
              {code && <span className="card-code mono">{code}</span>}
              {title && (
                <h2 className="card-title">
                  {Icon ? <Icon className="card-title-icon" aria-hidden="true" /> : null}
                  <span>{title}</span>
                </h2>
              )}
            </div>
            {subtitle && <p className="card-subtitle">{subtitle}</p>}
          </div>
          {action && <div className="card-action">{action}</div>}
        </header>
      )}
      {children}
    </section>
  );
}
