import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Smart Wheelchair Dashboard",
  description: "Real-time sensor monitoring for smart wheelchair",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
