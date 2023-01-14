import type { NextPage } from "next";
import Head from "next/head";
import styles from "../styles/Home.module.css";
import ImageCanvas from "../components/ImageCanvas";

const Home: NextPage = () => {
  return (
    <div className={styles.container}>
      <Head>
        <title>Theme conversion web playground</title>
        <meta name="description" content="Generated by create next app" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <main className={styles.main}>
        <h1 className={styles.title}>Theme conversion web playground</h1>

        <ImageCanvas className="inputImage" width={240} height={240} />

        <div id="result" className="mt-3"></div>
      </main>

      <footer className={styles.footer}>
        <a
          href="https://onnxruntime.ai/docs"
          target="_blank"
          rel="noopener noreferrer"
        ></a>
      </footer>
    </div>
  );
};

export default Home;