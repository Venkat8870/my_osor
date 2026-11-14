import type { Metadata } from 'next';
// @ts-ignore Next.js processes this stylesheet import at build time.
import './global.css';

export const metadata: Metadata = {
  title: 'Dukaan Saathi',
  description: 'Your intelligent shop assistant',
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