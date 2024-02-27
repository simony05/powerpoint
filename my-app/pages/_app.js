import { SessionProvider } from "next-auth/react"
import Navbar from '../components/navbar'
import '../styles/navbar.css'

function App({ Component, pageProps }) {
  return (
      <SessionProvider session={pageProps.session}>
        <Navbar></Navbar>
        <Component {...pageProps} />
      </SessionProvider>
  );
}
 
 export default App;