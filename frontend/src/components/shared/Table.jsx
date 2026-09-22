import './Table.css';

export function Table({ children, caption, ...props }) {
  return (
    <div className="table-wrapper" role="region" aria-label={caption} tabIndex={0}>
      <table className="table" {...props}>
        {caption && <caption className="table__caption sr-only">{caption}</caption>}
        {children}
      </table>
    </div>
  );
}

export function Thead({ children }) {
  return <thead className="table__head">{children}</thead>;
}

export function Tbody({ children }) {
  return <tbody className="table__body">{children}</tbody>;
}

export function Th({ children, align = 'left', ...props }) {
  return (
    <th className={`table__th table__th--${align}`} scope="col" {...props}>
      {children}
    </th>
  );
}

export function Td({ children, align = 'left', mono = false, ...props }) {
  return (
    <td
      className={['table__td', `table__td--${align}`, mono ? 'table__td--mono' : ''].filter(Boolean).join(' ')}
      {...props}
    >
      {children}
    </td>
  );
}

export function Tr({ children, ...props }) {
  return <tr className="table__tr" {...props}>{children}</tr>;
}
