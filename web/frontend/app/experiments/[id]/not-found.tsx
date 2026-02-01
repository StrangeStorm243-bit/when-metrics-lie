import Link from "next/link";

export default function NotFound() {
  return (
    <div style={{ textAlign: "center", padding: "4rem 2rem" }}>
      <h1>Experiment Not Found</h1>
      <p style={{ color: "#666", marginTop: "1rem" }}>
        The experiment you're looking for doesn't exist or has no results yet.
      </p>
      <Link
        href="/"
        style={{
          display: "inline-block",
          marginTop: "2rem",
          padding: "0.5rem 1rem",
          backgroundColor: "#0070f3",
          color: "white",
          borderRadius: "4px",
          textDecoration: "none",
        }}
      >
        Back to Home
      </Link>
    </div>
  );
}

